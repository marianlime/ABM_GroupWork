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

        # --- GRAPHICS SETUP ---
        # Initialize the Matplotlib figure and link it to the PySide6 canvas
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # Information label at the bottom for real-time feedback
        self.info_label = QLabel("DSP Analysis: Waiting for hover...")
        
        # Standard vertical layout for UI components
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.info_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Run the simulation logic once on startup
        self.apply_dsp_and_plot()

    def assign_precision(self, num_agents, dist_type="bimodal"):
        """
        SIMULATION STEP 1: Assign 'Skill Levels' to Agents.
        We create a bimodal distribution: half the agents are very accurate (low sigma),
        and the other half are very noisy (high sigma).
        """
        if dist_type == "uniform":
            return np.random.uniform(0.1, 0.5, num_agents)
        elif dist_type == "bimodal":
            # Group A: Accurate 'Professional' traders (Noise ~0.1)
            group_a = np.random.normal(0.1, 0.05, num_agents // 2)
            # Group B: Erratic 'Retail' traders (Noise ~0.9)
            group_b = np.random.normal(0.9, 0.05, num_agents // 2)
            return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)
        elif dist_type == "skewed":
            return np.random.lognormal(-1, 0.5, num_agents)

    def apply_dsp_and_plot(self):
        """
        SIMULATION STEP 2: The DSP Engine.
        This generates raw noisy signals and uses a filter to 'clean' them.
        """
        self.ax.clear()
        num_agents = 80
        true_shock = 1.2  # The actual underlying market value change
        
        # Get the noise profile for our 80 agents
        sigmas = self.assign_precision(num_agents, dist_type="bimodal")
        
        # DSP FILTER: A 3-point Moving Average Kernel
        # This averages the last 3 data points to reduce the impact of random spikes.
        window_size = 3
        kernel = np.ones(window_size) / window_size
        
        filtered_results = []
        for i in range(num_agents):
            s = sigmas[i]
            # Generate 50 points of raw, log-normal noise centered around 0
            agent_raw_noise = np.random.lognormal(0, s, 50) - np.exp(s**2 / 2)
            
            # MATH STEP: Convolution
            # This 'slides' our 3-point kernel over the noisy data to smooth it.
            filtered = np.convolve(agent_raw_noise, kernel, mode='valid')
            
            # Final Agent Signal = True Shock + The most recent filtered noise point
            filtered_results.append(true_shock + filtered[-1]) 
            
        # X-Axis: The refined signal (the price the agent thinks is correct)
        self.agent_x = np.array(filtered_results)
        # Y-Axis: Random Jitter (Spreads dots out so they don't overlap on a flat line)
        self.agent_y = np.random.normal(0, 0.5, num_agents)
        
        # --- PLOTTING ---
        self.ax.set_xlim(-1, 3); self.ax.set_ylim(-2, 2); self.ax.set_aspect('equal')
        self.ax.set_xlabel("Price Signal (After DSP Filtering)")
        self.ax.set_ylabel("Agent Distribution (Jitter)")
        
        # Draw the "Decision Zones" (The Venn-style circles)
        # Red = Sentiment for Selling; Green = Sentiment for Buying
        self.ax.add_patch(patches.Circle((0.3, 0), 1.0, color='red', alpha=0.15, label='Sell'))
        self.ax.add_patch(patches.Circle((1.7, 0), 1.0, color='green', alpha=0.15, label='Buy'))

        # Scatter plot for agents
        self.scatter = self.ax.scatter(self.agent_x, self.agent_y, c='#2c3e50', s=50, edgecolors='w', picker=True)
        
        # Connect the mouse-movement event to our hover function
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.draw()

    def on_hover(self, event):
        """
        INTERACTION STEP: Trading Logic.
        When you hover over a dot, this checks the filtered signal against 
        pre-defined thresholds to decide the agent's action.
        """
        if event.inaxes == self.ax:
            # Check if the mouse is currently touching one of the scatter dots
            cont, ind = self.scatter.contains(event)
            if cont:
                idx = ind["ind"][0] # Get the specific agent ID
                val = self.agent_x[idx] # Get their specific DSP-filtered signal
                
                # TRADING STRATEGY LOGIC:
                # Based on the filtered value, categorize the trader's action
                if val > 1.8: status, color = "STRONG BUY", "#27ae60"
                elif val > 1.3: status, color = "WEAK BUY", "#2ecc71"
                elif val < 0.2: status, color = "STRONG SELL", "#c0392b"
                elif val < 0.7: status, color = "WEAK SELL", "#e74c3c"
                else: status, color = "HOLD / NEUTRAL", "#7f8c8d"

                # Update UI label with dynamic styling
                self.info_label.setText(f"Agent {idx} | Decision: {status}")
                self.info_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            else:
                # Reset label if mouse is not hovering over a dot
                self.info_label.setText("Hover over an agent to see their signal impact.")
                self.info_label.setStyleSheet("color: #2c3e50; font-size: 14px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DSPVennSignal()
    window.show()
    sys.exit(app.exec())