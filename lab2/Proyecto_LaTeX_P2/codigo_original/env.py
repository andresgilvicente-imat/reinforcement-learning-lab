"""Entorno "Jump To The Goal" (Aprendizaje por Refuerzo · IMAT).

Fichero común a la Práctica 1 y a la Práctica 2: es el mismo entorno, y las dos
prácticas reparten una copia idéntica de este fichero.

Grid world de 7x5 inspirado en el Flappy Bird. El jugador solo puede avanzar en
diagonal hacia la derecha y debe alcanzar la celda objetivo evitando los
obstáculos.

Convenciones del enunciado
--------------------------
* Estados: las 20 celdas en las que el agente puede encontrarse, numeradas de
  izquierda a derecha y de arriba a abajo (s0 ... s19), más **un único estado
  terminal T** (índice 20). En total, nS = 21.
* Al chocar con un obstáculo y al alcanzar el objetivo se transita al mismo
  estado T; lo que distingue ambos casos es la recompensa de la transición
  (r_obstacle frente a r_goal), no el estado de destino.
* Acciones: {0: arriba-derecha, 1: abajo-derecha}.
* El entorno solo tiene borde por la derecha: desde la última columna cualquier
  acción hace caer al jugador una celda por debajo, en la misma columna
  (s3 -> s7 con las dos acciones).
* Recompensas: ``r_step`` en cualquier transición entre estados no terminales,
  ``r_obstacle`` al chocar y ``r_goal`` al alcanzar el objetivo.
* Estados iniciales: s_init = {s0, s4, s8, s12} (primera columna).

En modo determinista la acción elegida se ejecuta con probabilidad 1. En modo
estocástico se ejecuta con probabilidad ``prob`` y la contraria con
``1 - prob``.

La rejilla 7x5 se mantiene internamente para construir el modelo y para dibujar
el entorno, pero **no es el espacio de estados**: la correspondencia entre
estados y celdas está en ``STATE_TO_CELL`` y ``CELL_TO_STATE``.

Este fichero NO debe modificarse durante la práctica.
"""

from io import StringIO

import numpy as np
from gymnasium import Env, spaces, utils

# --- Acciones -------------------------------------------------------------
UP_RIGHT = 0
DOWN_RIGHT = 1

# --- Mapa: X = obstáculo, F = celda libre, G = objetivo -------------------
GRID_WORLD = [
    "XXXXX",
    "FFXFF",
    "FFXFF",
    "FFFFG",
    "FXFFF",
    "FXFFF",
    "XXXXX",
]


def _build_state_numbering(grid):
    """Numera s0 ... sN-1 las celdas visitables, en orden de lectura."""
    ncol = len(grid[0])
    cell_to_state, state_to_cell = {}, {}
    for row, line in enumerate(grid):
        for col, content in enumerate(line):
            if content == "F":
                cell = row * ncol + col
                cell_to_state[cell] = len(state_to_cell)
                state_to_cell[cell_to_state[cell]] = cell
    return cell_to_state, state_to_cell


CELL_TO_STATE, STATE_TO_CELL = _build_state_numbering(GRID_WORLD)

# --- Espacio de estados ---------------------------------------------------
N_NON_TERMINAL = len(STATE_TO_CELL)  # 20 estados visitables
TERMINAL_STATE = N_NON_TERMINAL  # T, único estado terminal
N_STATES = N_NON_TERMINAL + 1  # 21

# --- Parámetros por defecto del enunciado ---------------------------------
# s_init = {s0, s4, s8, s12}: las celdas de la primera columna desde las que
# existe camino al objetivo. La quinta, s16, se excluye a propósito: sus dos
# acciones caen en obstáculo, luego V*(s16) = r_obs con cualquier gamma y r_s.
INITIAL_STATES = (0, 4, 8, 12)

R_STEP = -1.0  # r_s
R_OBSTACLE = -5.0  # r_obs
R_GOAL = 5.0  # r_goal

# --- Colores del render (RGB) ---------------------------------------------
COLOR_FREE = (79, 172, 234)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_GOAL = (6, 160, 27)
COLOR_AGENT = (255, 143, 0)
COLOR_GRID = (180, 200, 230)
CELL_PIXELS = 100


class JumpToTheGoalEnv(Env):
    """Entorno Gymnasium tabular con modelo de transiciones ``P`` accesible.

    Args:
        render_mode (str | None): ``"human"`` (ventana pygame), ``"ansi"``
            (texto) o ``None`` (sin render).
        deterministic (bool): si ``True`` la acción elegida siempre se ejecuta.
        prob (float): en modo estocástico, probabilidad de ejecutar la acción
            elegida.
        r_step (float): recompensa r_s de una transición a un estado no terminal.
        r_obstacle (float): recompensa r_obs al chocar con un obstáculo.
        r_goal (float): recompensa r_goal al alcanzar el objetivo.
        initial_state (int | None): estado inicial fijo. Si es ``None`` se
            muestrea uniformemente de :data:`INITIAL_STATES` en cada ``reset``.

    Atributos relevantes para la práctica:
        nS (int): número de estados (21: s0 ... s19 y el terminal T).
        nA (int): número de acciones (2).
        P (dict): ``P[s][a] -> [(prob, next_state, reward, terminal), ...]``.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        deterministic=True,
        prob=1.0,
        r_step=R_STEP,
        r_obstacle=R_OBSTACLE,
        r_goal=R_GOAL,
        initial_state=INITIAL_STATES[0],
    ):
        super().__init__()
        self.desc = np.asarray(GRID_WORLD, dtype="c")
        self.nrow, self.ncol = self.desc.shape

        self.deterministic = deterministic
        self.prob = 1.0 if deterministic else prob
        self.r_step = float(r_step)
        self.r_obstacle = float(r_obstacle)
        self.r_goal = float(r_goal)

        self.nA = 2
        self.nS = N_STATES

        self.initial_state = initial_state
        self.state = self._sample_initial_state()
        # Celda del grid en la que está el agente. Solo se usa para dibujar:
        # como todas las celdas terminales son el mismo estado T, el estado por
        # sí solo no basta para saber dónde pintar al agente.
        self.cell = STATE_TO_CELL[self.state]
        self.lastaction = None

        self.P, self._next_cells = self._build_transition_model()

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # ------------------------------------------------------------- modelo
    def _next_cell(self, row, col, action):
        """Aplica la dinámica de movimiento del enunciado.

        El entorno solo tiene borde por la derecha: si el movimiento se sale
        por la última columna, el jugador cae una celda por debajo en la misma
        columna, sea cual sea la acción.
        """
        new_row = row - 1 if action == UP_RIGHT else row + 1
        new_col = col + 1

        new_row = min(max(new_row, 0), self.nrow - 1)
        if new_col > self.ncol - 1:
            new_row = min(row + 1, self.nrow - 1)
            new_col = self.ncol - 1
        return new_row * self.ncol + new_col

    def _transition(self, cell, action):
        """Devuelve ``(next_state, reward, terminal, next_cell)``."""
        row, col = divmod(cell, self.ncol)
        next_cell = self._next_cell(row, col, action)
        letter = self.desc[divmod(next_cell, self.ncol)]

        if letter == b"X":
            return TERMINAL_STATE, self.r_obstacle, True, next_cell
        if letter == b"G":
            return TERMINAL_STATE, self.r_goal, True, next_cell
        return CELL_TO_STATE[next_cell], self.r_step, False, next_cell

    def _build_transition_model(self):
        """Construye ``P[s][a] -> [(prob, next_state, reward, terminal), ...]``.

        Devuelve además la lista paralela de celdas de destino, que el render
        necesita para saber en qué obstáculo concreto ha acabado el episodio.
        El estado terminal T es absorbente: se queda en sí mismo con
        recompensa 0.
        """
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        next_cells = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for state in range(N_NON_TERMINAL):
            cell = STATE_TO_CELL[state]
            for a in range(self.nA):
                intentos = [a] if self.deterministic else list(range(self.nA))
                for b in intentos:
                    if self.deterministic:
                        p = 1.0
                    else:
                        p = self.prob if b == a else 1.0 - self.prob
                    next_state, reward, terminated, next_cell = self._transition(
                        cell, b
                    )
                    P[state][a].append((p, next_state, reward, terminated))
                    next_cells[state][a].append(next_cell)

        for a in range(self.nA):
            P[TERMINAL_STATE][a].append((1.0, TERMINAL_STATE, 0.0, True))
            next_cells[TERMINAL_STATE][a].append(None)

        return P, next_cells

    # ---------------------------------------------------------------- api
    def _sample_initial_state(self):
        """Estado inicial fijo o muestreado de ``INITIAL_STATES``."""
        if self.initial_state is None:
            return int(self.np_random.choice(INITIAL_STATES))
        return int(self.initial_state)

    def reset(self, seed=None, options=None):
        """Reinicia el episodio.

        ``options={"initial_state": s}`` fuerza el estado inicial del episodio.
        """
        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            self.state = int(options["initial_state"])
        else:
            self.state = self._sample_initial_state()
        self.cell = STATE_TO_CELL[self.state]
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.state), {"prob": 1.0}

    def step(self, action):
        """Ejecuta una acción y devuelve ``(obs, reward, terminated, truncated, info)``."""
        transitions = self.P[self.state][action]
        if len(transitions) == 1:
            i = 0
        else:
            probabilidades = [t[0] for t in transitions]
            i = int(self.np_random.choice(len(transitions), p=probabilidades))
        p, s, r, terminated = transitions[i]

        next_cell = self._next_cells[self.state][action][i]
        if next_cell is not None:
            self.cell = next_cell
        self.state = int(s)
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        return int(s), r, bool(terminated), False, {"prob": p}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "human":
            return self._render_gui()
        return None

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    # ------------------------------------------------------------- render
    def _render_text(self):
        """Render en texto: la celda del agente aparece resaltada."""
        out = StringIO()
        desc = [[c.decode("utf-8") for c in line] for line in self.desc.tolist()]
        row, col = divmod(self.cell, self.ncol)
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        out.write("\n".join("".join(line) for line in desc) + "\n")
        return out.getvalue()

    def _render_gui(self):
        """Render con pygame: una ventana con el grid y el agente."""
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("pygame no está instalado: `pip install pygame`") from exc

        width = self.ncol * CELL_PIXELS
        height = self.nrow * CELL_PIXELS

        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Jump to the Goal")
            self.window = pygame.display.set_mode((width, height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((255, 255, 255))

        for row in range(self.nrow):
            for col in range(self.ncol):
                rect = pygame.Rect(
                    (col * CELL_PIXELS, row * CELL_PIXELS), (CELL_PIXELS, CELL_PIXELS)
                )
                letter = self.desc[row, col]
                if letter == b"X":
                    color = COLOR_OBSTACLE
                elif letter == b"G":
                    color = COLOR_GOAL
                else:
                    color = COLOR_FREE
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, COLOR_GRID, rect, 1)

        agent_row, agent_col = divmod(self.cell, self.ncol)
        pygame.draw.circle(
            canvas,
            COLOR_AGENT,
            (
                agent_col * CELL_PIXELS + CELL_PIXELS // 2,
                agent_row * CELL_PIXELS + CELL_PIXELS // 2,
            ),
            CELL_PIXELS // 3,
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
