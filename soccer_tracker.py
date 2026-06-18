import numpy as np
import cv2
import math

class SoccerTracker:
    def __init__(self, fps=30.0, cm_per_pixel=1.0, possession_threshold=50.0, 
                 yellow_goal_roi=None, blue_goal_roi=None, team_mapping=None, debug=False):
        """
        Clase para calcular estadísticas de fútbol robot
        
        Args:
            fps (float): Cuadros por segundo (FPS) del vídeo.
            cm_per_pixel (float): Factor de escala físico (cm por píxel).
            possession_threshold (float): Distancia máxima (en píxeles) para considerar posesión de balón.
            yellow_goal_roi (list): ROI de la portería amarilla.
                - Polígono de 4 puntos: [x1, y1, x2, y2, x3, y3, x4, y4]
            blue_goal_roi (list): ROI de la portería azul.
                - Polígono de 4 puntos: [x1, y1, x2, y2, x3, y3, x4, y4]
            team_mapping (dict): Mapeo de obj_id a nombre del equipo (e.g. {1: 'Team Yellow', 2: 'Team Blue'}).
        """
        self.fps = fps
        self.cm_per_pixel = cm_per_pixel
        self.possession_threshold = possession_threshold
        
        # Orden de puntos para ROI poligonal: arriba-izquierda, arriba-derecha,
        # abajo-derecha, abajo-izquierda (sentido horario).
        self.yellow_goal_roi = self._normalize_goal_roi(
            yellow_goal_roi if yellow_goal_roi is not None else [0, 0, 0, 0]
        )
        self.blue_goal_roi = self._normalize_goal_roi(
            blue_goal_roi if blue_goal_roi is not None else [0, 0, 0, 0]
        )
        
        self.team_mapping = team_mapping if team_mapping is not None else {}
        self.debug = debug
        
        self.ball_trajectory = [] # Lista de (frame_idx, bx, by)
        self.robot_trajectories = {} # obj_id -> Lista de (frame_idx, rx, ry)
        self.robot_velocities = {} # obj_id -> Lista de (frame_idx, speed_cm_s)
        
        self.possession_history = [] # Lista de (frame_idx, obj_id, team_name)
        self.possession_counts = {"Team Yellow": 0, "Team Blue": 0, "Loose": 0}
        
        self.goals_scored = []
        self.in_goal_state = None # None, 'yellow' o 'blue'
        self.goal_cooldown_counter = 0
        self.cooldown_frames = int(fps * 2.0) # 2 segundos de cooldown para re-evaluar goles
        
        self.scores = {"Team Yellow": 0, "Team Blue": 0}

        # 4 robots
        self.tracker_id_map = {}
        self.last_frame_raw_to_pid = {}
        self.last_selected_ball_box = None
        self.current_frame_idx = 0
        
        # IDs de cada equipo 
        self.yellow_pids = []
        self.blue_pids = []
        for pid, team_name in self.team_mapping.items():
            if "yellow" in team_name.lower():
                self.yellow_pids.append(pid)
            elif "blue" in team_name.lower():
                self.blue_pids.append(pid)

        # 2 IDs por equipo
        all_defined_pids = set(self.yellow_pids + self.blue_pids)
        next_candidate = 1
        
        while len(self.yellow_pids) < 2:
            if next_candidate not in all_defined_pids:
                self.yellow_pids.append(next_candidate)
                all_defined_pids.add(next_candidate)
                self.team_mapping[next_candidate] = f"Robot Yellow {next_candidate}"
            next_candidate += 1
            
        while len(self.blue_pids) < 2:
            if next_candidate not in all_defined_pids:
                self.blue_pids.append(next_candidate)
                all_defined_pids.add(next_candidate)
                self.team_mapping[next_candidate] = f"Robot Blue {next_candidate}"
            next_candidate += 1

    def _get_center_from_box(self, box):
        if box is None or len(box) < 4:
            return None
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def _point_inside_box(self, px, py, box, margin=0.0):
        if box is None or len(box) < 4:
            return False
        x, y, w, h = box
        return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)

    def _normalize_goal_roi(self, roi):
        if roi is None:
            return [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

        if isinstance(roi, np.ndarray):
            roi = roi.tolist()

        # Caso 1: Lista de 4 puntos [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        if len(roi) == 4 and isinstance(roi[0], (list, tuple, np.ndarray)):
            return [(float(pt[0]), float(pt[1])) for pt in roi]

        # Caso 2: 4 números planos [xmin, ymin, xmax, ymax]
        if len(roi) == 4:
            xmin, ymin, xmax, ymax = [float(v) for v in roi]
            return [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]

        # Caso 3: 8 números planos [x1, y1, x2, y2, x3, y3, x4, y4]
        if len(roi) == 8:
            return [
                (float(roi[0]), float(roi[1])),
                (float(roi[2]), float(roi[3])),
                (float(roi[4]), float(roi[5])),
                (float(roi[6]), float(roi[7])),
            ]

        raise ValueError("goal_roi debe tener 4 puntos (x,y) o una lista de 4 u 8 valores planos")

    def _goal_center_y(self, goal_roi):
        return float(np.mean([p[1] for p in goal_roi]))

    def _point_inside_goal_roi(self, px, py, goal_roi):
        contour = np.array(goal_roi, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0

    def _calculate_iou(self, box1, box2):
        """Calcula el Intersection over Union (IoU) y el Intersection over Area (IoA)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Coordenadas de las esquinas
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
        
        # Intersección
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
            
        iou = inter_area / union_area
        ioa = inter_area / min(box1_area, box2_area)
        return max(iou, ioa)

    def _filter_duplicate_robot_boxes(self, robot_boxes_dict, masks_dict=None, overlap_threshold=0.6):
        """
        Filtra cajas duplicadas/superpuestas que representan el mismo robot
        """
        if not robot_boxes_dict:
            return {}, {} if masks_dict is not None else None

        filtered_boxes = {}
        filtered_masks = {} if masks_dict is not None else None
        
        for key in sorted(robot_boxes_dict.keys()):
            box = robot_boxes_dict[key]
            
            #si se superpone demasiado con alguna caja
            is_duplicate = False
            for accepted_box in filtered_boxes.values():
                if self._calculate_iou(box, accepted_box) > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_boxes[key] = box
                if masks_dict is not None and key in masks_dict:
                    filtered_masks[key] = masks_dict[key]
                    
        return filtered_boxes, filtered_masks

    def _is_ball_candidate_valid(self, ball_box, robot_boxes_dict, robot_margin_ratio=0.12, max_area_ratio=0.35):
        if ball_box is None or len(ball_box) < 4:
            return False

        x, y, w, h = [float(v) for v in ball_box[:4]]
        if w <= 0 or h <= 0:
            return False

        ball_center = (x + w / 2.0, y + h / 2.0)
        ball_area = w * h

        robot_areas = []
        for robot_box in robot_boxes_dict.values():
            if robot_box is None or len(robot_box) < 4:
                continue

            rx, ry, rw, rh = [float(v) for v in robot_box[:4]]
            robot_areas.append(rw * rh)

            margin_x = rw * robot_margin_ratio
            margin_y = rh * robot_margin_ratio
            expanded_box = [rx - margin_x, ry - margin_y, rw + 2.0 * margin_x, rh + 2.0 * margin_y]
            if self._point_inside_box(ball_center[0], ball_center[1], expanded_box):
                return False

        if robot_areas:
            reference_robot_area = min(robot_areas)
            if ball_area > reference_robot_area * max_area_ratio:
                return False

        return True

    def _select_ball_candidate(self, ball_candidates, robot_boxes_dict):
        if ball_candidates is None:
            return None

        if isinstance(ball_candidates, (list, tuple, np.ndarray)) and len(ball_candidates) > 0 and isinstance(ball_candidates[0], (list, tuple, np.ndarray)):
            candidates = [box for box in ball_candidates if self._is_ball_candidate_valid(box, robot_boxes_dict)]
            if not candidates:
                return None

            if len(self.ball_trajectory) == 0:
                return candidates[0]

            _, last_bx, last_by = self.ball_trajectory[-1]
            best_box = candidates[0]
            min_dist = float('inf')
            for box in candidates:
                center = self._get_center_from_box(box)
                if center is None:
                    continue
                bx, by = center
                dist = math.sqrt((bx - last_bx)**2 + (by - last_by)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_box = box
            return best_box

        return ball_candidates if self._is_ball_candidate_valid(ball_candidates, robot_boxes_dict) else None
    
    def _map_robot_detections(self, robot_boxes_dict, masks_dict=None, update_state=True):
        """
        Mapea IDs cambiantes/arbitrarios del tracker a un conjunto persistente:
        - IDs para el equipo Yellow (self.yellow_pids)
        - IDs para el equipo Blue (self.blue_pids)
        """
        robot_boxes_dict, masks_dict = self._filter_duplicate_robot_boxes(robot_boxes_dict, masks_dict)

        mapped_boxes = {}
        mapped_masks = {} if masks_dict is not None else None

        if update_state:
            already_mapped_pids = set()
            unmapped_raw_ids = []
            
            active_pid_to_raw = {}
            for raw_id in robot_boxes_dict:
                if raw_id in self.tracker_id_map:
                    pid = self.tracker_id_map[raw_id]
                    if pid not in active_pid_to_raw:
                        active_pid_to_raw[pid] = raw_id
                        mapped_boxes[pid] = robot_boxes_dict[raw_id]
                        if masks_dict is not None and raw_id in masks_dict:
                            mapped_masks[pid] = masks_dict[raw_id]
                        already_mapped_pids.add(pid)
                    else:
                        # Conflicto detectado, forzar reasignación
                        unmapped_raw_ids.append(raw_id)
                else:
                    unmapped_raw_ids.append(raw_id)

            # Calcular el centro de las porterías
            yellow_center_y = self._goal_center_y(self.yellow_goal_roi)
            blue_center_y = self._goal_center_y(self.blue_goal_roi)

            # primer ID desocupado de su respectivo equipo
            for raw_id in unmapped_raw_ids:
                box = robot_boxes_dict[raw_id]
                center = self._get_center_from_box(box)
                if center is None:
                    continue
                rx, ry = center
                
                dist_to_yellow = abs(ry - yellow_center_y)
                dist_to_blue = abs(ry - blue_center_y)
                
                if dist_to_yellow < dist_to_blue:
                    # Equipo Yellow
                    pid = None
                    for y_pid in self.yellow_pids:
                        if y_pid not in already_mapped_pids:
                            pid = y_pid
                            break
                    if pid is None:
                        continue # Ambos IDs de Yellow ocupados en este frame
                else:
                    # Equipo Blue
                    pid = None
                    for b_pid in self.blue_pids:
                        if b_pid not in already_mapped_pids:
                            pid = b_pid
                            break
                    if pid is None:
                        continue # Ambos IDs de Blue ocupados en este frame

                self.tracker_id_map[raw_id] = pid
                mapped_boxes[pid] = box
                if masks_dict is not None and raw_id in masks_dict:
                    mapped_masks[pid] = masks_dict[raw_id]
                already_mapped_pids.add(pid)
                
            self.last_frame_raw_to_pid = {raw_id: pid for raw_id, pid in self.tracker_id_map.items() if raw_id in robot_boxes_dict}
        else:
            last_mapping = getattr(self, 'last_frame_raw_to_pid', {})
            active_pid_to_raw = {}
            for raw_id, box in robot_boxes_dict.items():
                pid = last_mapping.get(raw_id)
                if pid is None:
                    pid = self.tracker_id_map.get(raw_id)
                
                if pid is not None and pid not in active_pid_to_raw:
                    active_pid_to_raw[pid] = raw_id
                    mapped_boxes[pid] = box
                    if masks_dict is not None and raw_id in masks_dict:
                        mapped_masks[pid] = masks_dict[raw_id]

        return mapped_boxes, mapped_masks

    def update(self, frame_idx, ball_box, robot_boxes_dict):
        """
        Actualiza el estado del juego con la información del frame actual.
        
        Args:
            frame_idx (int): Índice del frame actual.
            ball_box (list): Caja delimitadora [x, y, w, h] del balón o None.
            robot_boxes_dict (dict): Diccionario obj_id -> [x, y, w, h] de los robots.
        """
        self.current_frame_idx = frame_idx

        # máximo 4
        robot_boxes_dict, _ = self._map_robot_detections(robot_boxes_dict, update_state=True)

        if isinstance(ball_box, list) and len(ball_box) == 0:
            ball_box = None

        ball_box = self._select_ball_candidate(ball_box, robot_boxes_dict)

        self.last_selected_ball_box = ball_box

        ball_center = self._get_center_from_box(ball_box)
        if ball_center is not None:
            bx, by = ball_center
            self.ball_trajectory.append((frame_idx, bx, by))
            
            if self.in_goal_state is not None:
                self.goal_cooldown_counter -= 1
                if self.goal_cooldown_counter <= 0:
                    in_yellow = self._point_inside_goal_roi(bx, by, self.yellow_goal_roi)
                    in_blue = self._point_inside_goal_roi(bx, by, self.blue_goal_roi)
                    if not in_yellow and not in_blue:
                        self.in_goal_state = None
            else:
                if self._point_inside_goal_roi(bx, by, self.yellow_goal_roi):
                    self.in_goal_state = 'yellow'
                    self.goal_cooldown_counter = self.cooldown_frames
                    goal_time = frame_idx / self.fps
                    self.goals_scored.append({
                        "team": "Team Blue", 
                        "time": goal_time, 
                        "frame": frame_idx
                    })
                    self.scores["Team Blue"] += 1
                
                elif self._point_inside_goal_roi(bx, by, self.blue_goal_roi):
                    self.in_goal_state = 'blue'
                    self.goal_cooldown_counter = self.cooldown_frames
                    goal_time = frame_idx / self.fps
                    self.goals_scored.append({
                        "team": "Team Yellow", 
                        "time": goal_time, 
                        "frame": frame_idx
                    })
                    self.scores["Team Yellow"] += 1

        for obj_id, box in robot_boxes_dict.items():
            rc = self._get_center_from_box(box)
            if rc is None:
                continue
            rx, ry = rc
            
            if obj_id not in self.robot_trajectories:
                self.robot_trajectories[obj_id] = []
                self.robot_velocities[obj_id] = []
            
            self.robot_trajectories[obj_id].append((frame_idx, rx, ry))
            
            history = self.robot_trajectories[obj_id]
            if len(history) >= 5:
                prev_frame, prev_x, prev_y = history[-5]
                dist_px = math.sqrt((rx - prev_x)**2 + (ry - prev_y)**2)
                dt = (frame_idx - prev_frame) / self.fps
                if dt > 0:
                    speed_physical = (dist_px * self.cm_per_pixel) / dt
                    self.robot_velocities[obj_id].append((frame_idx, speed_physical))
            else:
                self.robot_velocities[obj_id].append((frame_idx, 0.0))

        if ball_center is not None and len(robot_boxes_dict) > 0:
            bx, by = ball_center
            min_dist = float('inf')
            possession_owner = None
            possession_team = "Loose"
            
            for obj_id, box in robot_boxes_dict.items():
                rc = self._get_center_from_box(box)
                if rc is None:
                    continue
                rx, ry = rc
                dist = math.sqrt((bx - rx)**2 + (by - ry)**2)
                if dist < min_dist:
                    min_dist = dist
                    possession_owner = obj_id
            
            if min_dist <= self.possession_threshold:
                team_name = self.team_mapping.get(possession_owner, f"Robot {possession_owner}")
                if "yellow" in team_name.lower():
                    possession_team = "Team Yellow"
                elif "blue" in team_name.lower():
                    possession_team = "Team Blue"
                else:
                    possession_team = team_name
            else:
                possession_team = "Loose"
                
            self.possession_history.append((frame_idx, possession_owner, possession_team))
            
            if possession_team in self.possession_counts:
                self.possession_counts[possession_team] += 1
            else:
                self.possession_counts[possession_team] = 1
        else:
            self.possession_history.append((frame_idx, None, "Loose"))
            self.possession_counts["Loose"] += 1

    def get_statistics(self):
        total_frames = sum(self.possession_counts.values())
        possession_pct = {}
        
        if total_frames > 0:
            for k, v in self.possession_counts.items():
                possession_pct[k] = (v / total_frames) * 100.0
        else:
            possession_pct = {"Team Yellow": 0.0, "Team Blue": 0.0, "Loose": 100.0}
            
        robot_speeds = {}
        for obj_id, speeds in self.robot_velocities.items():
            if len(speeds) > 0:
                speed_vals = [s[1] for s in speeds]
                robot_speeds[obj_id] = {
                    "avg_speed": float(np.mean(speed_vals)),
                    "max_speed": float(np.max(speed_vals)),
                    "team": self.team_mapping.get(obj_id, f"Robot {obj_id}")
                }
        
        return {
            "possession_seconds": {k: v / self.fps for k, v in self.possession_counts.items()},
            "possession_percentage": possession_pct,
            "goals": self.goals_scored,
            "final_score": self.scores,
            "robot_speeds": robot_speeds
        }

    def generate_heatmap(self, bg_image, kernel_size=51, alpha=0.5):
        """
        Genera y superpone un mapa de calor del balón la imagen de fondo.
        
        Args:
            bg_image (np.ndarray): Imagen de fondo (cancha de fútbol).
            kernel_size (int): Tamaño del kernel Gaussiano para suavizar la densidad.
            alpha (float): Transparencia del mapa de calor overlay (0.0 a 1.0).
            
        Returns:
            np.ndarray: Imagen final con el mapa de calor integrado.
        """
        h, w, c = bg_image.shape
        heatmap_accumulation = np.zeros((h, w), dtype=np.float32)
        
        for _, bx, by in self.ball_trajectory:
            ix, iy = int(round(bx)), int(round(by))
            if 0 <= ix < w and 0 <= iy < h:
                # Agregar un incremento
                heatmap_accumulation[iy, ix] += 1.0
                
        if np.max(heatmap_accumulation) == 0:
            return bg_image.copy()
            
        # efecto de calor continuo
        heatmap_blurred = cv2.GaussianBlur(heatmap_accumulation, (kernel_size, kernel_size), 0)
        
        # Normalizar [0, 255]
        heatmap_normalized = np.uint8(255 * (heatmap_blurred / np.max(heatmap_blurred)))
        
        # azul (baja densidad) rojo (alta densidad))
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # máscara de el balón para no colorear toda la cancha de azul
        mask = (heatmap_blurred > 0.005 * np.max(heatmap_blurred))
        
        blended = bg_image.copy()
        blended[mask] = cv2.addWeighted(bg_image, 1.0 - alpha, heatmap_color, alpha, 0)[mask]
        
        return blended

    def annotate_frame(self, frame, ball_box, robot_boxes_dict, masks_dict=None):
        """
        Dibuja los recuadros, máscaras, velocidades, marcadores y posesión sobre el frame actual.
        
        Args:
            frame (np.ndarray): Frame original.
            ball_box (list): [x, y, w, h] del balón o None.
            robot_boxes_dict (dict): obj_id -> [x, y, w, h] de robots.
            masks_dict (dict): obj_id -> máscara binaria o None.
            
        Returns:
            np.ndarray: Frame anotado.
        """

        # máximo 4
        robot_boxes_dict, masks_dict = self._map_robot_detections(robot_boxes_dict, masks_dict, update_state=False)

        ball_box = self._select_ball_candidate(ball_box, robot_boxes_dict)

        annotated = frame.copy()
        h, w, _ = annotated.shape
        
        # Dibujar ROIs de porterías si debug
        if getattr(self, 'debug', False):
            overlay = annotated.copy()
            if self.yellow_goal_roi and len(self.yellow_goal_roi) == 4:
                yellow_pts = np.array(self.yellow_goal_roi, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [yellow_pts], (0, 255, 255))
                cv2.polylines(annotated, [yellow_pts], isClosed=True, color=(0, 255, 255), thickness=2)
                x1 = int(np.min(yellow_pts[:, :, 0]))
                y1 = int(np.min(yellow_pts[:, :, 1]))
                cv2.putText(annotated, "Goal Yellow ROI", (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            if self.blue_goal_roi and len(self.blue_goal_roi) == 4:
                blue_pts = np.array(self.blue_goal_roi, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [blue_pts], (255, 120, 0))
                cv2.polylines(annotated, [blue_pts], isClosed=True, color=(255, 120, 0), thickness=2)
                x1 = int(np.min(blue_pts[:, :, 0]))
                y1 = int(np.min(blue_pts[:, :, 1]))
                cv2.putText(annotated, "Goal Blue ROI", (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1, cv2.LINE_AA)
            annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
        
        # Dibujar estela de 3 segundos
        max_trail_frames = int(3.0 * self.fps)
        current_frame = getattr(self, 'current_frame_idx', 0)
        min_frame_idx = current_frame - max_trail_frames

        ball_pts = [ (bx, by, f_idx) for f_idx, bx, by in self.ball_trajectory if f_idx >= min_frame_idx ]
        if len(ball_pts) > 1:
            for i in range(len(ball_pts) - 1):
                p1 = (int(round(ball_pts[i][0])), int(round(ball_pts[i][1])))
                p2 = (int(round(ball_pts[i+1][0])), int(round(ball_pts[i+1][1])))
                age = current_frame - ball_pts[i][2]
                alpha = max(0.1, min(1.0, 1.0 - (age / max_trail_frames)))
                
                color = (0, 0, int(round(255 * alpha)))
                thickness = max(1, int(round(3 * alpha)))
                cv2.line(annotated, p1, p2, color, thickness, cv2.LINE_AA)


        
        if masks_dict is not None:
            for obj_id, mask in masks_dict.items():
                if mask is not None:
                    # color por equipo
                    team_name = self.team_mapping.get(obj_id, f"Robot {obj_id}")
                    if "yellow" in team_name.lower():
                        color = (0, 255, 255) 
                    elif "blue" in team_name.lower():
                        color = (255, 0, 0) 
                    elif obj_id == 0 or "ball" in team_name.lower():
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0) 
                        
                    if not isinstance(mask, np.ndarray):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask
                    mask_np = np.squeeze(mask_np)
                        
                    color_mask = np.zeros_like(annotated)
                    color_mask[mask_np > 0] = color
                    annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.4, 0)

        if ball_box is not None:
            x, y, wb, hb = [int(v) for v in ball_box]
            cv2.rectangle(annotated, (x, y), (x + wb, y + hb), (0, 0, 255), 2)
            cv2.putText(annotated, "Ball", (x, max(0, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for obj_id, box in robot_boxes_dict.items():
            x, y, wr, hr = [int(v) for v in box]
            
            team_name = self.team_mapping.get(obj_id, f"Robot {obj_id}")
            if "yellow" in team_name.lower():
                color = (0, 255, 255) # Amarillo
            elif "blue" in team_name.lower():
                color = (255, 120, 0) # Azul
            else:
                color = (0, 255, 0)
                
            cv2.rectangle(annotated, (x, y), (x + wr, y + hr), color, 2)
            
            current_speed = 0.0
            if obj_id in self.robot_velocities and len(self.robot_velocities[obj_id]) > 0:
                current_speed = self.robot_velocities[obj_id][-1][1]
                
            label = f"ID:{obj_id} {current_speed:.1f} cm/s"
            cv2.putText(annotated, label, (x, max(0, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        #Scoreboard
        cv2.rectangle(annotated, (10, 10), (220, 80), (40, 40, 40), -1)
        cv2.rectangle(annotated, (10, 10), (220, 80), (200, 200, 200), 1)
        
        cv2.putText(annotated, "MARCADOR ROBOT-FUT", (18, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        score_text = f"Yellow: {self.scores['Team Yellow']} | Blue: {self.scores['Team Blue']}"
        cv2.putText(annotated, score_text, (18, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Posesión del balón 
        current_possession = "Loose"
        if len(self.possession_history) > 0:
            current_possession = self.possession_history[-1][2]
            
        poss_text = f"Posesion: {current_possession}"
        if current_possession == "Team Yellow":
            poss_color = (0, 255, 255)
        elif current_possession == "Team Blue":
            poss_color = (255, 200, 0)
        else:
            poss_color = (200, 200, 200)
            
        cv2.putText(annotated, poss_text, (18, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, poss_color, 1, cv2.LINE_AA)

        if self.in_goal_state is not None and self.goal_cooldown_counter > self.cooldown_frames - 15:
            text = "¡¡¡ G O O O L !!!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            tx = (w - text_size[0]) // 2
            ty = (h + text_size[1]) // 2
            
            cv2.rectangle(annotated, (tx - 10, ty - text_size[1] - 10), (tx + text_size[0] + 10, ty + 10), (0, 0, 0), -1)
            cv2.putText(annotated, text, (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255) if self.in_goal_state == 'blue' else (255, 120, 0), 3, cv2.LINE_AA)

        return annotated
