# In src/models/data_augmentation_v2/visualization/training_plots.py
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingVisualizer:
    def plot_losses(self, generator_losses: List[float], critic_losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(generator_losses, label='Generator Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.title('Training Losses Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# In src/models/data_augmentation_v2/visualization/evaluation_plots.py
class EvaluationVisualizer:
    def plot_distributions(self, real_data: np.ndarray, generated_data: np.ndarray):
        """Plot distribution comparisons"""
        pass

    def plot_metrics(self, metrics: Dict[str, List[float]]):
        """Plot evaluation metrics"""
        pass