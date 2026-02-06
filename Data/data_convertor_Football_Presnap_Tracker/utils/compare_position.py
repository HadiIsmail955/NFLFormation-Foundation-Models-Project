def compare_position(player_a, player_b):
    center_x_a = player_a["bbox"][0] + player_a["bbox"][2] / 2
    center_x_b = player_b["bbox"][0] + player_b["bbox"][2] / 2

    if center_x_a < center_x_b:
        return -1
        # print("Player A is to the LEFT of Player B")
    elif center_x_a > center_x_b:
        return 1
        # print("Player A is to the RIGHT of Player B")
    else:
        return 0
        # print("Players are aligned on the same x-axis")