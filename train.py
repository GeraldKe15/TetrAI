from agent import DQAgent
from tetrai import Tetrai
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import os

env = Tetrai()

batch_size = 512
episodes = 210
epochs = 10
max_steps = None
train_every = 10
render_every = 1

agent = DQAgent()

scores = []
average_scores = []

for episode in tqdm(range(episodes)):
    render = False
    current_state = env.reset()
    done = False
    steps = 0

    current_state = env.reset()

    if episode % render_every == 0:
        render = True

    while not done and (not max_steps or steps < max_steps):

        next_states = env.get_next_states()
        best_pos, best_rot, best_state = agent.policy(next_states)
        score, done = env.play(best_pos, best_rot, render)

        agent.append_memory(
            current_state, (best_pos, best_rot), score, best_state, done)
        current_state = best_state
        steps += 1

    scores.append(env.get_score())

    if episode % train_every == 0:
        print("training")
        agent.train(batch_size=batch_size, epochs=epochs)

    # outputs the average score of the last 50 plays
    if episode % 50 == 0:
        last50 = scores[-50:] if episode > 0 else []
        last50_score = sum(last50)/50
        average_scores.append(last50_score)
        print(episode, last50_score)

# plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.plot(range(0, episodes, 50), average_scores, label='Average Score')
plt.title('Average Scores After 200 Episodes')
plt.xlabel('# of Episodes')
plt.ylabel('Average Score')
plt.legend()

model_save_path = f"results/tetrai_model_{timestamp}"
tf.saved_model.save(agent.model, model_save_path)
img_save_path = os.path.join(model_save_path, 'score_plot.png')
plt.savefig(img_save_path)
print("Trained model saved at:", model_save_path)

plt.show()
