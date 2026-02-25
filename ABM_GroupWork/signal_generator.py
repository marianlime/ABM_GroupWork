import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


def assign_noise_parameter_set(num_agents, dist_type="bimodal"):
    if dist_type == "uniform":
        return np.random.uniform(0.1, 0.5, num_agents)
    elif dist_type == "bimodal":
        group_a = np.random.normal(0.1, 0.05, num_agents // 2)
        group_b = np.random.normal(0.9, 0.05, num_agents // 2)
        return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)
    elif dist_type == "skewed":
        return np.random.lognormal(-1, 0.5, num_agents)

def signal_generator(noise_parameter, S_next, noise_distribution='lognormal'):
    if noise_distribution == 'lognormal':
        return S_next * np.exp(np.random.normal(0, noise_parameter))
    if noise_distribution == 'uniform':
        return 0
    




class DSPVennSignal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP-Driven Agent Signal Analysis")
        self.setMinimumSize(800, 750)

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        self.info_label = QLabel("DSP Analysis: Waiting for hover...")
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.info_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.apply_dsp_and_plot()

    def assign_precision(self, num_agents, dist_type="bimodal"):
        if dist_type == "uniform":
            return np.random.uniform(0.1, 0.5, num_agents)
        elif dist_type == "bimodal":
            group_a = np.random.normal(0.1, 0.05, num_agents // 2)
            group_b = np.random.normal(0.9, 0.05, num_agents // 2)
            return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)
        elif dist_type == "skewed":
            return np.random.lognormal(-1, 0.5, num_agents)

    def apply_dsp_and_plot(self):
        self.ax.clear()
        num_agents = 80
        true_shock = 1.2 
        
        sigmas = self.assign_precision(num_agents, dist_type="bimodal")
        
        window_size = 3
        kernel = np.ones(window_size) / window_size
        
        filtered_results = []
        for i in range(num_agents):
            s = sigmas[i]
            agent_raw_noise = np.random.lognormal(0, s, 50) - np.exp(s**2 / 2)
            filtered = np.convolve(agent_raw_noise, kernel, mode='valid')
            filtered_results.append(true_shock + filtered[-1]) 
            
        self.agent_x = np.array(filtered_results)
        self.agent_y = np.random.normal(0, 0.5, num_agents)
        
        self.ax.set_xlim(-1, 3); self.ax.set_ylim(-2, 2); self.ax.set_aspect('equal')
        
        self.ax.add_patch(patches.Circle((0.3, 0), 1.0, color='red', alpha=0.15, label='Sell'))
        self.ax.add_patch(patches.Circle((1.7, 0), 1.0, color='green', alpha=0.15, label='Buy'))

        self.scatter = self.ax.scatter(self.agent_x, self.agent_y, c='#2c3e50', s=50, edgecolors='w', picker=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                val = self.agent_x[idx]
                
                if val > 1.8: status, color = "STRONG BUY", "#27ae60"
                elif val > 1.3: status, color = "WEAK BUY", "#2ecc71"
                elif val < 0.2: status, color = "STRONG SELL", "#c0392b"
                elif val < 0.7: status, color = "WEAK SELL", "#e74c3c"
                else: status, color = "HOLD / NEUTRAL", "#7f8c8d"

                self.info_label.setText(f"Agent {idx} | Decision: {status}")
                self.info_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            else:
                self.info_label.setText("Hover over an agent to see their signal impact.")
                self.info_label.setStyleSheet("color: #2c3e50; font-size: 14px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DSPVennSignal()
    window.show()
    sys.exit(app.exec())