# This program is for solving the simple vaccume promplem using simle reflex agent Reinforcement learning.
# Context - We have 4 rooms. Some rooms are dirty and some rooms are clean. 
# The vacume cleaner will visit each room. If the room dirty, cleans it and moves to next room. If the room is clean moves to next room.
# Since the vaccume has no memory, he will visit each room to make sure each room is clean.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import msvcrt


# Defining the 2X2 environment
environment = {
    "Room1": "Clean",
    "Room2": "Dirty", # Start with dirt room from this position
    "Room3": "Clean",
    "Room4": "Clean",
}

# Mapping for grid positions
room_positions = {
    "Room1": (0, 1), # Top-Left
    "Room2": (1, 1), # Top-Right
    "Room3": (0, 0), # Bottom-Left
    "Room4": (1, 0), # Bottom-Right
}

rooms = list(environment.keys())

#reflex Agent Function
def reflex_agent(room_state):
    if room_state == "Dirty":
        return "Clean"
    else:
        return "Move"

#function to draw the Grid
def draw_environment(environment_details, agent_position, current_step):
    fig, axis_x = plt.subplots()
    axis_x.set_xlim(0,2)
    axis_x.set_ylim(0,2)
    axis_x.set_xticks([])
    axis_x.set_yticks([])
    axis_x.set_title(f"Step {current_step} - Agent in {rooms[agent_position]}")

    for room, pos in room_positions.items():
        x_axis_position, y_axis_position = pos
        room_color = 'red' if environment_details [room] == "Dirty" else 'green'
        rectangle = patches.Rectangle((x_axis_position,y_axis_position),1,1, facecolor=room_color, edgecolor='black')
        axis_x.add_patch(rectangle)
        axis_x.text(x_axis_position + 0.5, y_axis_position + 0.5, room, ha='center', va='center', color='white', fontsize=10)

    # Draw agent
    agent_x, agent_y = room_positions[rooms[agent_position]]        #f finds the agent's current position
    agent_patch = patches.Circle ((agent_x + 0.5, agent_y + 0.5), 0.1, color='blue') # Blue circle
    axis_x.add_patch(agent_patch)

    plt.pause(10)
    plt.close()


# Main method
def main():
    plt.ion()
    total_steps = 8
    agent_index = 0

    # Loop throgu each step and check the room status, take action accordingly
    for current_step in range(total_steps):
        current_room = rooms[agent_index]
        state = environment[current_room]
        action = reflex_agent(state)

        # Calling the draw environment method with current agent index and next step
        draw_environment(environment, agent_index, current_step + 1)

        if action == "Clean" : # If the agent decides to clean, this line updates
            environment[current_room] = "Clean"
        else:
            agent_index = (agent_index + 1) % len(rooms)            
            
    plt.ioff()
    print("Simulation complete!")

 # Calling the meain method for running the simulation
if __name__ == "__main__":
    main()