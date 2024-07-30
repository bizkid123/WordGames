import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import numpy as np

def read_file(file_path):
	with open(file_path, 'r') as file:
		lines = file.readlines()
	
	data = []
	for i in range(0, len(lines), 6):
		step = [x.upper() for x in [lines[i].strip(), lines[i+1].strip(), lines[i+2].strip(), lines[i+3].strip(), lines[i+4].strip()]]
		data.append(step)
	return data

def create_images(data, highlight = False):
	images = []
	tile_size = 0.75
	gap = 0.3
	grid_size = 4 * (tile_size + gap)  # Total size of the grid
	
	prev = data[0]
	data = [prev] + data
	for prev,step in zip(data, data[1:]):
		fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size for square format
		ax.set_xlim(0, grid_size)
		ax.set_ylim(0, grid_size)
		
		# Draw grid and fill cells with more gap and rounded corners
		for i in range(4):
			for j in range(4):
				x = j * (tile_size + gap) + gap/2
				y = (3 - i) * (tile_size + gap) + gap/2

				color = "#d3a86e" if (step[i][j] == prev[i][j]) else "#e8c78f"

				rect = patches.FancyBboxPatch((x, y), tile_size, tile_size, 
											  boxstyle="round,pad=0.1", linewidth=2, edgecolor='black', facecolor=color)
				ax.add_patch(rect)
				ax.text(x + tile_size / 2, y + tile_size / 2, (prev, step)[highlight][i][j], ha='center', va='center', fontsize=32, weight='bold', color='black')
		
		ax.axis('off')
		plt.title(f"Score: {(prev, step)[highlight][4]}", fontsize=20, weight='bold', pad=20)  # Increase font size and weight for the title
		plt.tight_layout()

		# Save image to buffer
		fig.canvas.draw()
		image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
		images.append(image)
		plt.close(fig)
	return images

def create_last_step_image(data, output_path):
	tile_size = 0.75
	gap = 0.3
	grid_size = 4 * (tile_size + gap)  # Total size of the grid
	
	last_step = data[-1]
	
	fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size for square format
	ax.set_xlim(0, grid_size)
	ax.set_ylim(0, grid_size)
	
	# Draw grid and fill cells with more gap and rounded corners
	for i in range(4):
		for j in range(4):
			x = j * (tile_size + gap) + gap/2
			y = (3 - i) * (tile_size + gap) + gap/2

			color = "#d3a86e"

			rect = patches.FancyBboxPatch((x, y), tile_size, tile_size, 
										  boxstyle="round,pad=0.1", linewidth=2, edgecolor='black', facecolor=color)
			ax.add_patch(rect)
			ax.text(x + tile_size / 2, y + tile_size / 2, last_step[i][j], ha='center', va='center', fontsize=32, weight='bold', color='black')
	
	ax.axis('off')
	plt.title(f"Score: {last_step[4]}", fontsize=20, weight='bold', pad=20)  # Increase font size and weight for the title
	plt.tight_layout()

	plt.savefig(output_path)
	plt.close(fig)

def create_gif(images, output_path):
	# Ensure each image is in RGB format
	rgb_images = [image[:, :, :3] for image in images]
	duration = [250] * (len(rgb_images)-1) + [4000]
	imageio.mimsave(output_path, rgb_images, format='GIF', duration=duration, loop=0)

def main():
	input_file = 'hill_climb_output2.txt'
	output_file = 'hill_climb_animation.gif'
	
	data = read_file(input_file)
	images1 = create_images(data)
	images2 = create_images(data, True)
	images = [item for pair in zip(images1, images2) for item in pair]

	create_gif(images, output_file)

if __name__ == "__main__":
	main()
	create_last_step_image(read_file("optimal_board.txt"), "best_board_new.png")