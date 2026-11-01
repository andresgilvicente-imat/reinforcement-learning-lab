"""Entorno de navegación continuo (Práctica 3 · Sesión 1 · RL IMAT).

Convenciones del enunciado:

* Recinto cuadrado de 10 x 10 m. El agente es un círculo de radio 0,2 m que se
  desplaza 0,25 m en la dirección elegida.
* Acciones: 0 = arriba, 1 = abajo, 2 = izquierda, 3 = derecha.
* Tres obstáculos verticales de 0,2 x 5 m con la esquina inferior izquierda en
  (1,9, 1), (4,9, 1) y (7,9, 1).
* Zona objetivo: rectángulo de 1 x 2 m con la esquina inferior izquierda en
  (2,5, 8).
* El episodio termina (``terminated``) si el agente choca con el perímetro o
  con un obstáculo (fracaso) o si su centro entra en la zona objetivo (éxito).
  Además se trunca (``truncated``) al llegar a ``max_steps`` pasos.
* Posición inicial aleatoria y uniforme entre las posiciones libres, fuera de
  la zona objetivo. ``reset(options={"posicion": (x, y)})`` la fija.
* Observación «cruda»: ``(x, y, colision, objetivo)``. Las dos últimas son
  0 o 1. Al agente no le llega esto, sino su codificación (``representacion.py``).
* Recompensa: la calcula la función que se pasa en ``recompensa``; por defecto,
  ``recompensa.recompensa``, que diseñáis vosotros.

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("arriba", "abajo", "izquierda", "derecha")

WIDTH = 10.0
HEIGHT = 10.0
# (x, y, ancho, alto) de cada obstáculo
OBSTACLES = (
    (1.9, 1.0, 0.2, 5.0),
    (4.9, 1.0, 0.2, 5.0),
    (7.9, 1.0, 0.2, 5.0),
)
TARGET_AREA = (2.5, 8.0, 1.0, 2.0)
AGENT_RADIUS = 0.2
AGENT_STEP = 0.25
MAX_STEPS = 2000

# Desplazamiento (dx, dy) de cada acción
MOVES = {
    UP: (0.0, AGENT_STEP),
    DOWN: (0.0, -AGENT_STEP),
    LEFT: (-AGENT_STEP, 0.0),
    RIGHT: (AGENT_STEP, 0.0),
}


def is_collision(x: float, y: float) -> bool:
    """``True`` si el agente centrado en ``(x, y)`` toca una pared u obstáculo."""
    r = AGENT_RADIUS
    if x <= r or x >= WIDTH - r or y <= r or y >= HEIGHT - r:
        return True
    return any(
        ox - r <= x <= ox + ow + r and oy - r <= y <= oy + oh + r
        for ox, oy, ow, oh in OBSTACLES
    )


def in_target(x: float, y: float) -> bool:
    """``True`` si el centro del agente está dentro de la zona objetivo."""
    tx, ty, tw, th = TARGET_AREA
    return tx <= x <= tx + tw and ty <= y <= ty + th


def draw_environment(ax) -> None:
    """Dibuja recinto, obstáculos y zona objetivo sobre unos ejes de matplotlib."""
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    for ox, oy, ow, oh in OBSTACLES:
        ax.add_patch(
            Rectangle((ox, oy), ow, oh, facecolor="indianred", edgecolor="darkred")
        )
    tx, ty, tw, th = TARGET_AREA
    ax.add_patch(
        Rectangle(
            (tx, ty), tw, th, facecolor="lightgreen", edgecolor="green", alpha=0.6
        )
    )


class NavigationEnv(gym.Env):
    """Navegación en un recinto continuo con obstáculos.

    Args:
        recompensa (callable | None): función
            ``recompensa(pos, nueva_pos, colision, objetivo) -> float``.
            Si es ``None`` se usa ``recompensa.recompensa``.
        max_steps (int): pasos a partir de los cuales se trunca el episodio.
        render_mode (str | None): ``"human"``, ``"rgb_array"`` o ``None``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, recompensa=None, max_steps=MAX_STEPS, render_mode=None):
        super().__init__()
        if recompensa is None:
            from recompensa import recompensa
        self.recompensa = recompensa
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.agent_pos = (0.0, 0.0)
        self.collision = False
        self.target = False
        self.steps = 0
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------ api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and options.get("posicion") is not None:
            x, y = (float(v) for v in options["posicion"])
            if is_collision(x, y) or in_target(x, y):
                raise ValueError(f"Posición inicial no válida: ({x}, {y})")
            self.agent_pos = (x, y)
        else:
            self.agent_pos = self._sample_free_position()
        self.collision = False
        self.target = False
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        if action not in MOVES:
            raise ValueError(f"Acción no válida: {action}")
        self.steps += 1
        pos = self.agent_pos
        dx, dy = MOVES[action]
        new_pos = (pos[0] + dx, pos[1] + dy)

        if is_collision(*new_pos):
            # El agente se queda donde estaba: el episodio acaba en fracaso.
            self.collision = True
        else:
            self.agent_pos = new_pos
            self.target = in_target(*new_pos)

        terminated = self.collision or self.target
        truncated = (not terminated) and self.steps >= self.max_steps
        reward = float(self.recompensa(pos, new_pos, self.collision, self.target))
        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------- internos
    def _get_obs(self):
        return np.array(
            [*self.agent_pos, float(self.collision), float(self.target)],
            dtype=np.float32,
        )

    def _sample_free_position(self):
        rng = self.np_random
        while True:
            x = float(rng.uniform(AGENT_RADIUS, WIDTH - AGENT_RADIUS))
            y = float(rng.uniform(AGENT_RADIUS, HEIGHT - AGENT_RADIUS))
            if not is_collision(x, y) and not in_target(x, y):
                return x, y

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()
        draw_environment(ax)
        ax.add_patch(Circle(self.agent_pos, AGENT_RADIUS, facecolor="orange"))
        ax.set_title(f"Paso {self.steps}")

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return rgba[..., :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


if __name__ == "__main__":
    # Diez pasos aleatorios con una recompensa nula, solo para ver el entorno.
    env = NavigationEnv(recompensa=lambda *_: 0.0, render_mode="human")
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        print(f"obs = {obs}  terminated = {terminated}")
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
