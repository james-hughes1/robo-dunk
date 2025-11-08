import math
import random
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces


@dataclass
class RoboDunkConfig:
    # Game
    fps: int = 60
    max_episode_steps: int = 1000

    # Difficulty Parameters
    bucket_height_min: int = 10
    bucket_height_max: int = 50
    bucket_width_min: int = 70
    bucket_width_max: int = 100
    bucket_y_min: int = 100
    bucket_y_max: int = 200
    arm_length_min: int = 40
    arm_length_max: int = 80
    ball_freq_min: int = 200
    ball_freq_max: int = 300

    # Rewards
    proximity_reward: float = 10.0
    time_penalty: float = 0.01


class RoboDunkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        self.config = config or RoboDunkConfig()
        self.render_mode = render_mode
        self.screen_width = 400
        self.screen_height = 400
        self.clock = pygame.time.Clock()
        self.fps = self.config.fps
        self.max_episode_steps = self.config.max_episode_steps

        self.action_space = spaces.MultiBinary(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )

        # Robot
        self.robot_speed = 5
        self.robot_width = 50
        self.robot_height = 20
        self.arm_min = 180
        self.arm_max = 360
        self.arm_speed = 4
        self.arm_elasticity = 2.0

        # Ball
        self.ball_radius = 8
        self.shoot_min_speed = 400
        self.shoot_max_speed = 600
        self.max_ball_speed = 700
        self.ball_elasticity = 1.3
        self.drag = 0.99

        # Cannon
        self.cannon_width = 6
        self.cannon_height = 30
        self.cannon_angle_min = 35
        self.cannon_angle_max = 60

        # General
        self.object_elasticity = 0.3
        self.difficulty = 0.0
        self.set_difficulty(self.difficulty)

        # Reward
        self.proximity_reward = self.config.proximity_reward
        self.time_penalty = self.config.time_penalty

        self._setup()

    def seed(self, seed=None):
        """Seed RNGs for reproducible behavior."""
        self._seed = seed
        self.np_random, seed_ = gym.utils.seeding.np_random(seed)
        random.seed(seed)  # python random
        self.set_difficulty(self.difficulty)
        return [seed_]

    def _setup(self):
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Robo Dunk Bucket")
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None

        self._elapsed_steps = 0
        self._ball_steps = 0

        self.space = pymunk.Space()
        self.space.gravity = (0, 400)
        self.static_body = self.space.static_body

        # Walls
        walls = [
            pymunk.Segment(self.static_body, (0, 0), (0, self.screen_height), 2),
            pymunk.Segment(
                self.static_body,
                (self.screen_width, 0),
                (self.screen_width, self.screen_height),
                2,
            ),
            pymunk.Segment(self.static_body, (0, 0), (self.screen_width, 0), 2),
        ]
        for wall in walls:
            wall.elasticity = self.object_elasticity
            wall.friction = 0.5
        self.space.add(*walls)

        # Bucket
        self.bucket_floor = pymunk.Segment(
            self.static_body,
            (self.screen_width - self.bucket_width, self.bucket_y),
            (self.screen_width, self.bucket_y),
            3,
        )
        self.bucket_wall = pymunk.Segment(
            self.static_body,
            (self.screen_width - self.bucket_width, self.bucket_y),
            (self.screen_width - self.bucket_width, self.bucket_y - self.bucket_height),
            3,
        )
        self.bucket_floor.elasticity = self.object_elasticity
        self.bucket_wall.elasticity = self.object_elasticity
        self.space.add(self.bucket_floor, self.bucket_wall)

        # Robot
        self.robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.robot_body.position = (self.screen_width // 4, self.screen_height - 50)
        self.robot_shape = pymunk.Poly.create_box(
            self.robot_body, (self.robot_width, self.robot_height)
        )
        self.robot_shape.elasticity = self.object_elasticity
        self.space.add(self.robot_body, self.robot_shape)

        self.arm_angle = self.arm_min
        self.arm_shape = self._create_arm()
        self.space.add(self.arm_shape)

        # Cannon
        self.cannon_base = (self.screen_width - 40, self.screen_height - 50)
        self.balls = []
        self.max_balls = 3

        self.score = 0

    def _create_arm(self):
        start = (
            self.robot_body.position.x,
            self.robot_body.position.y - self.robot_height / 2,
        )
        end = (
            start[0] + self.arm_length * math.cos(math.radians(-self.arm_angle)),
            start[1] - self.arm_length * math.sin(math.radians(-self.arm_angle)),
        )
        arm = pymunk.Segment(self.static_body, start, end, 5)
        arm.elasticity = self.arm_elasticity
        arm.friction = 0.5
        return arm

    def _spawn_ball(self):
        # Use self.np_random if available, else fallback to random
        rng = getattr(self, "np_random", np.random)

        angle = rng.integers(self.cannon_angle_min, self.cannon_angle_max + 1)
        speed = rng.integers(self.shoot_min_speed, self.shoot_max_speed + 1)
        rad = math.radians(angle)

        tip_x = self.cannon_base[0] - self.cannon_height * math.cos(rad)
        tip_y = self.cannon_base[1] - self.cannon_height * math.sin(rad)

        ball_body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, self.ball_radius))
        ball_body.position = tip_x, tip_y
        ball_shape = pymunk.Circle(ball_body, self.ball_radius)
        ball_shape.elasticity = self.ball_elasticity
        ball_shape.friction = 0.5

        ball_body.velocity = (-speed * math.cos(rad), -speed * math.sin(rad))
        ball_body.velocity_func = (
            lambda body, gravity, damping, dt: pymunk.Body.update_velocity(
                body, gravity, self.drag, dt
            )
        )
        self._ball_steps = 0
        self.next_cannon_angle = angle
        self.space.add(ball_body, ball_shape)
        self.balls.append((ball_body, ball_shape))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._setup()
        return self._get_obs(), {}

    def _get_obs(self):
        # Render the current state to an offscreen pygame surface
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((255, 255, 255))

        # Draw all objects (use same draw options as render)
        self.space.debug_draw(pymunk.pygame_util.DrawOptions(surface))

        # Convert pygame surface (W,H,3) → numpy array (H,W,3)
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))

        # Convert RGB to grayscale (still high-res, no resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Add channel dimension → (H, W, 1)
        obs = np.expand_dims(gray, axis=-1).astype(np.uint8)

        return obs

    def step(self, action):
        self._elapsed_steps += 1

        # --- Handle Robot Movement ---
        if action[0]:
            self.robot_body.position = (
                self.robot_body.position.x - self.robot_speed,
                self.robot_body.position.y,
            )
        if action[1]:
            self.robot_body.position = (
                self.robot_body.position.x + self.robot_speed,
                self.robot_body.position.y,
            )
        if action[2]:
            self.arm_angle = min(self.arm_max, self.arm_angle + self.arm_speed)
        if action[3]:
            self.arm_angle = max(self.arm_min, self.arm_angle - self.arm_speed)

        # Keep robot within bounds
        self.robot_body.position = (
            max(
                self.robot_width // 2,
                min(
                    self.screen_width // 2 - self.robot_width // 2,
                    self.robot_body.position.x,
                ),
            ),
            self.robot_body.position.y,
        )

        # Update arm
        self.space.remove(self.arm_shape)
        self.arm_shape = self._create_arm()
        self.space.add(self.arm_shape)

        # Step simulation
        self.space.step(1 / self.fps)

        reward = 0
        terminated = False

        for body, shape in self.balls:
            # --- Clamp ball speed ---
            # Clamp ball speed
            vx, vy = body.velocity
            speed = (vx**2 + vy**2) ** 0.5
            if speed > self.max_ball_speed:
                scale = self.max_ball_speed / speed
                body.velocity = vx * scale, vy * scale

            # --- Check if ball scored ---
            dist_to_bucket = self.bucket_floor.point_query(body.position).distance
            if dist_to_bucket <= self.ball_radius and abs(body.velocity.y) < 1:
                reward += 10
                self.score += 1
                self.space.remove(body, shape)
                self.balls.remove((body, shape))
            else:
                reward = np.exp(-dist_to_bucket / self.config.proximity_reward)

            # --- Penalize if ball hits ground ---
            if body.position.y >= self.screen_height - self.ball_radius - 1:
                reward -= 5  # strong penalty
                self.space.remove(body, shape)
                self.balls.remove((body, shape))

        if (self._ball_steps % self.ball_freq == 0 or (not self.balls)) and len(
            self.balls
        ) < self.max_balls:
            self._spawn_ball()
        self._ball_steps += 1

        # --- End episode if too long ---
        if self._elapsed_steps >= self.max_episode_steps:
            terminated = True

        # --- Time step penalty ---
        reward -= self.config.time_penalty

        reward = float(reward)
        return self._get_obs(), reward, terminated, False, {}

    def _set_difficulty_param(self, p_min, p_max, difficulty, reverse=False):
        rng = getattr(self, "np_random", np.random)
        difficulty_jump = rng.integers(0, int(difficulty * (p_max - p_min)) + 1)
        if not reverse:
            return p_min + difficulty_jump
        else:
            return p_max - difficulty_jump

    def set_difficulty(self, difficulty_level):
        self.difficulty = difficulty_level
        self.bucket_height = self._set_difficulty_param(
            self.config.bucket_height_min,
            self.config.bucket_height_max,
            difficulty_level,
        )
        self.bucket_width = self._set_difficulty_param(
            self.config.bucket_width_min,
            self.config.bucket_width_max,
            difficulty_level,
            reverse=True,
        )
        self.bucket_y = self._set_difficulty_param(
            self.config.bucket_y_min,
            self.config.bucket_y_max,
            difficulty_level,
            reverse=True,
        )
        self.arm_length = self._set_difficulty_param(
            self.config.arm_length_min,
            self.config.arm_length_max,
            difficulty_level,
            reverse=True,
        )
        self.ball_freq = self._set_difficulty_param(
            self.config.ball_freq_min,
            self.config.ball_freq_max,
            difficulty_level,
            reverse=True,
        )

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((255, 255, 255))

        # Conveyor belt
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, self.screen_height - 40),
            (self.screen_width // 2, self.screen_height - 40),
            3,
        )

        # Cannon
        angle = math.radians(self.next_cannon_angle)
        offset = 5
        tip_x = self.cannon_base[0] - self.cannon_height * math.cos(angle)
        tip_y = self.cannon_base[1] - self.cannon_height * math.sin(angle)
        left_bar_start = (
            self.cannon_base[0] - offset * math.sin(angle),
            self.cannon_base[1] + offset * math.cos(angle),
        )
        left_bar_end = (
            tip_x - offset * math.sin(angle),
            tip_y + offset * math.cos(angle),
        )
        right_bar_start = (
            self.cannon_base[0] + offset * math.sin(angle),
            self.cannon_base[1] - offset * math.cos(angle),
        )
        right_bar_end = (
            tip_x + offset * math.sin(angle),
            tip_y - offset * math.cos(angle),
        )
        pygame.draw.line(self.screen, (34, 139, 34), left_bar_start, left_bar_end, 3)
        pygame.draw.line(self.screen, (34, 139, 34), right_bar_start, right_bar_end, 3)

        # Physics objects
        self.space.debug_draw(self.draw_options)

        # --- Draw the score bar ---
        font = pygame.font.Font(None, 28)  # default font, size 28
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))  # black text
        self.screen.blit(text, (10, 10))  # top-left corner

        # Update Display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
