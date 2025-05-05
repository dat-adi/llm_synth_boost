from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import glob
import os

# Find the latest log file
log_files = glob.glob("./logs/events.out.tfevents.*")
if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    
    # Load the events
    ea = event_accumulator.EventAccumulator(latest_log)
    ea.Reload()
    print(ea.scalars.Keys())
    
    # Extract the training loss
    if 'eval/loss' in ea.scalars.Keys():
        loss_events = ea.scalars.Items('eval/loss')
        steps = [event.step for event in loss_events]
        losses = [event.value for event in loss_events]
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses)
        plt.title('Evaluation Loss vs. Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('evaluation_loss_plot.png')
        plt.show()
