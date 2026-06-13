import numpy as np
import cv2
import math

class SoccerTracker:
    def __init__(self, fps=30.0, cm_per_pixel=1.0, possession_threshold=50.0, 
                 yellow_goal_roi=None, blue_goal_roi=None, team_mapping=None):
        """
        Clase para calcular estadísticas de fútbol robot
        
        Args:
            fps (float): Cuadros por segundo (FPS) del vídeo.
            cm_per_pixel (float): Factor de escala físico (cm por píxel).
            possession_threshold (float): Distancia máxima (en píxeles) para considerar posesión de balón.
            yellow_goal_roi (list): [xmin, ymin, xmax, ymax] de la portería amarilla.
            blue_goal_roi (list): [xmin, ymin, xmax, ymax] de la portería azul.
            team_mapping (dict): Mapeo de obj_id a nombre del equipo (e.g. {1: 'Team Yellow', 2: 'Team Blue'}).
        """
        self.fps = fps
        self.cm_per_pixel = cm_per_pixel
        self.possession_threshold = possession_threshold
        
        self.yellow_goal_roi = yellow_goal_roi if yellow_goal_roi is not None else [0, 0, 0, 0]
        self.blue_goal_roi = blue_goal_roi if blue_goal_roi is not None else [0, 0, 0, 0]
        
        self.team_mapping = team_mapping if team_mapping is not None else {}
        
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

    def _get_center_from_box(self, box):
        if box is None or len(box) < 4:
            return None
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, frame_idx, ball_box, robot_boxes_dict):
        """
        Actualiza el estado del juego con la información del frame actual.
        
        Args:
            frame_idx (int): Índice del frame actual.
            ball_box (list): Caja delimitadora [x, y, w, h] del balón o None.
            robot_boxes_dict (dict): Diccionario obj_id -> [x, y, w, h] de los robots.
        """
        ball_center = self._get_center_from_box(ball_box)
        if ball_center is not None:
            bx, by = ball_center
            self.ball_trajectory.append((frame_idx, bx, by))
            
            if self.in_goal_state is not None:
                self.goal_cooldown_counter -= 1
                if self.goal_cooldown_counter <= 0:
                    in_yellow = (self.yellow_goal_roi[0] <= bx <= self.yellow_goal_roi[2] and 
                                 self.yellow_goal_roi[1] <= by <= self.yellow_goal_roi[3])
                    in_blue = (self.blue_goal_roi[0] <= bx <= self.blue_goal_roi[2] and 
                               self.blue_goal_roi[1] <= by <= self.blue_goal_roi[3])
                    if not in_yellow and not in_blue:
                        self.in_goal_state = None
            else:
                if (self.yellow_goal_roi[0] <= bx <= self.yellow_goal_roi[2] and 
                    self.yellow_goal_roi[1] <= by <= self.yellow_goal_roi[3]):
                    self.in_goal_state = 'yellow'
                    self.goal_cooldown_counter = self.cooldown_frames
                    goal_time = frame_idx / self.fps
                    self.goals_scored.append({
                        "team": "Team Blue", 
                        "time": goal_time, 
                        "frame": frame_idx
                    })
                    self.scores["Team Blue"] += 1
                
                elif (self.blue_goal_roi[0] <= bx <= self.blue_goal_roi[2] and 
                      self.blue_goal_roi[1] <= by <= self.blue_goal_roi[3]):
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
        annotated = frame.copy()
        h, w, _ = annotated.shape
        
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
