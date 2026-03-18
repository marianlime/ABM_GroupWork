import sys
import time
import random
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLineEdit, QFormLayout, QGroupBox)
from PySide6.QtCore import QThread, Signal

# ==========================================
# 1. THE BRAIN: Background Simulation Thread
# ==========================================
class SimulationWorker(QThread):
    # Signal sends: (Generation, Stats Dictionary)
    live_update = Signal(int, dict)
    finished = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def run(self):
        n_gens = int(self.params.get("n_generations", 50))
        zi_agents = int(self.params.get("n_zi", 35))
        informed_agents = int(self.params.get("n_informed", 65))
        
        # ---> YOUR DUCKDB 'BEGIN TRANSACTION' GOES HERE <---

        # Mock starting values for the evolution traits
        current_aggression = 1.0
        current_patience = 0.5

        for gen in range(n_gens):
            if not self.is_running:
                break 
            
            # ---> 1. YOUR ACTUAL play_game() CODE GOES HERE <---
            # g = play_game(n_zi_agents=zi_agents, n_parameterised_agents=informed_agents, ...)
            
            # --- MOCK DATA FOR TESTING THE GUI ---
            time.sleep(0.1) # Pretend it's doing heavy math
            
            # Simulate the traits evolving over time
            current_aggression += random.uniform(-0.05, 0.15) # Trending up
            current_patience += random.uniform(-0.02, 0.01)   # Trending down
            
            mock_stats = {
                "wealth_informed": 1000 + (gen * 15) + random.randint(-50, 50),
                "wealth_zi": 1000 + (gen * 5) + random.randint(-20, 20),
                "mean_aggression": current_aggression,
                "mean_patience": current_patience
            }

            # ---> 2. SEND THIS GENERATION'S DATA TO THE GUI <---
            self.live_update.emit(gen, mock_stats)

        # ---> YOUR DUCKDB 'COMMIT' GOES HERE <---
        self.finished.emit()

    def stop(self):
        self.is_running = False


# ==========================================
# 2. THE FACE: The PySide6 Dashboard
# ==========================================
class CommandCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABM Command Center")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: THE MENU ---
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        menu_group = QGroupBox("Simulation Parameters")
        menu_form = QFormLayout()
        
        # New Inputs
        self.input_gens = QLineEdit("100")
        self.input_zi = QLineEdit("35")
        self.input_informed = QLineEdit("65")
        self.input_volatility = QLineEdit("0.2")
        self.input_mutation = QLineEdit("0.05")
        
        menu_form.addRow("Generations:", self.input_gens)
        menu_form.addRow("ZI Agents:", self.input_zi)
        menu_form.addRow("Informed Agents:", self.input_informed)
        menu_form.addRow("GBM Volatility:", self.input_volatility)
        menu_form.addRow("Mutation Rate:", self.input_mutation)
        menu_group.setLayout(menu_form)
        
        left_layout.addWidget(menu_group)

        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setStyleSheet("background-color: #2e8b57; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.start_simulation)
        left_layout.addWidget(self.btn_run)
        
        left_layout.addStretch() 
        main_layout.addWidget(left_panel)

        # --- RIGHT PANEL: MULTIPLE GRAPHS ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.graph_area = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_area, stretch=1)

        # Window 1: Trait Evolution (Replacing the Scatter Plot)
        self.plot_traits = self.graph_area.addPlot(title="Evolution of Decision Making (Population Mean)")
        self.plot_traits.setLabel('bottom', 'Generation')
        self.plot_traits.setLabel('left', 'Trait Value')
        self.plot_traits.addLegend()
        
        self.line_aggression = self.plot_traits.plot(pen=pg.mkPen((255, 165, 0), width=3), name="Mean Aggression")
        self.line_patience = self.plot_traits.plot(pen=pg.mkPen((138, 43, 226), width=3), name="Mean Patience")

        self.graph_area.nextRow()

        # Window 2: Wealth Line Graph
        self.plot_wealth = self.graph_area.addPlot(title="Wealth Race: Informed vs ZI")
        self.plot_wealth.setLabel('bottom', 'Generation')
        self.plot_wealth.setLabel('left', 'Average Wealth ($)')
        self.plot_wealth.addLegend()
        
        self.line_informed_wealth = self.plot_wealth.plot(pen=pg.mkPen('b', width=2), name="Informed")
        self.line_zi_wealth = self.plot_wealth.plot(pen=pg.mkPen('r', width=2), name="ZI Agents")
        
        # Data storage lists
        self.history_gen = []
        self.history_agg = []
        self.history_pat = []
        self.history_w_inf = []
        self.history_w_zi = []

    def start_simulation(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("RUNNING...")
        
        # Clear old graph data
        self.history_gen.clear()
        self.history_agg.clear()
        self.history_pat.clear()
        self.history_w_inf.clear()
        self.history_w_zi.clear()
        
        # Grab inputs
        params = {
            "n_generations": self.input_gens.text(),
            "n_zi": self.input_zi.text(),
            "n_informed": self.input_informed.text(),
            "volatility": self.input_volatility.text(),
            "mutation": self.input_mutation.text()
        }
        
        self.worker = SimulationWorker(params)
        self.worker.live_update.connect(self.update_gui)
        self.worker.finished.connect(self.simulation_finished)
        self.worker.start()

    def update_gui(self, gen, stats):
        """Draws the new data point on the graphs every generation."""
        self.history_gen.append(gen)
        
        # Top Graph (Traits)
        self.history_agg.append(stats['mean_aggression'])
        self.history_pat.append(stats['mean_patience'])
        self.line_aggression.setData(self.history_gen, self.history_agg)
        self.line_patience.setData(self.history_gen, self.history_pat)

        # Bottom Graph (Wealth)
        self.history_w_inf.append(stats['wealth_informed'])
        self.history_w_zi.append(stats['wealth_zi'])
        self.line_informed_wealth.setData(self.history_gen, self.history_w_inf)
        self.line_zi_wealth.setData(self.history_gen, self.history_w_zi)

    def simulation_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("RUN SIMULATION")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CommandCenter()
    window.show()
    sys.exit(app.exec())