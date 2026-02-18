import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

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

    def apply_dsp_and_plot(self):
        self.ax.clear()
        num_agents = 80
        # BOOSTED True Shock to ensure we get out of the 'Hold' zone
        true_shock = 1.2 
        
        # --- DSP ENGINE ---
        # 1. Generate Raw Stochastic Signal (Lognormal as per your brief)
        # We generate a buffer of 50 samples for each agent
        sigma = 0.5
        raw_noise = np.random.lognormal(0, sigma, (num_agents, 50)) - np.exp(sigma**2 / 2)
        
        # 2. Apply Causal Moving Average Filter (DSP Kernel)
        window_size = 3 # Smaller window = less smoothing = more distinct signals
        kernel = np.ones(window_size) / window_size
        
        filtered_results = []
        for i in range(num_agents):
            # Convolution: This is the causal DSP filtering step
            filtered = np.convolve(raw_noise[i], kernel, mode='valid')
            # The final signal is the True Shock + the filtered noise
            filtered_results.append(true_shock + filtered[-1]) 
            
        self.agent_x = np.array(filtered_results)
        self.agent_y = np.random.normal(0, 0.5, num_agents) # Visual spread
        
        # --- VISUALIZATION ---
        self.ax.set_xlim(-1, 3); self.ax.set_ylim(-2, 2); self.ax.set_aspect('equal')
        
        # Adjusted Venn Zones for better "Signal Strength" mapping
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
                
                # RECALIBRATED THRESHOLDS
                # If value is far right -> Strong Buy. Far left -> Strong Sell.
                if val > 1.8: status, color = "STRONG BUY", "#27ae60"
                elif val > 1.3: status, color = "WEAK BUY", "#2ecc71"
                elif val < 0.2: status, color = "STRONG SELL", "#c0392b"
                elif val < 0.7: status, color = "WEAK SELL", "#e74c3c"
                else: status, color = "HOLD / NEUTRAL", "#7f8c8d"

                self.info_label.setText(f"Agent {idx} | DSP Output: {val:.3f} | Decision: {status}")
                self.info_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            else:
                self.info_label.setText("Hover over an agent to see their signal impact.")
                self.info_label.setStyleSheet("color: #2c3e50; font-size: 14px;")

    def assign_precision(self, num_agents, dist_type="uniform"):
        """Satisfies the 'Create a function to assign signal precision' requirement."""
        if dist_type == "uniform":
        # Every agent has a random noise level between 0.1 and 0.5
          return np.random.uniform(0.1, 0.5, num_agents)
    
        elif dist_type == "bimodal":
        # Half 'smart' agents (0.1 noise), half 'clueless' agents (0.9 noise)
          group_a = np.random.normal(0.1, 0.05, num_agents // 2)
          group_b = np.random.normal(0.9, 0.05, num_agents // 2)
          return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)

        elif dist_type == "skewed":
        # Most agents are noisy, only a few are precise (Log-normal distribution of noise)
          return np.random.lognormal(-1, 0.5, num_agents)

# Inside your generate_and_evaluate function:
        sigmas = assign_precision(num_agents, dist_type="bimodal")
        for i in range(num_agents):
        # Now each agent i uses their specific precision sigma[i]
          noise = np.random.lognormal(0, sigmas[i]) - np.exp(sigmas[i]**2 / 2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DSPVennSignal()
    window.show()
    sys.exit(app.exec())