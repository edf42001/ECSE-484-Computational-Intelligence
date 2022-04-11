import numpy as np
import cv2


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


data_file = "hidden_state_viz_data/hidden_data.csv"
text_file = "hidden_state_viz_data/generated_text.txt"
viz_file = "hidden_state.html"

with open(text_file, 'r') as f:
    generated_text = f.read()


data = np.loadtxt(data_file, delimiter=",")

print("Text: " + generated_text)
length = len(generated_text)

# cv2.imshow("Activations", data)
# cv2.waitKey()


# Node index to look at
node_index = 68

with open(viz_file, "w") as f:
    for i in range(length):
        node_activation = data[i, node_index]

        if node_activation < 0:
            rgb = (int(255 + 255 * node_activation), int(255 + 255 * node_activation), int(255))
        else:
            rgb = (int(255), int(255 + 255 * node_activation), int(255 + 255 * node_activation))

        # Debug node_activations->color mapping
        # print(rgb_to_hex(rgb), node_activation)

        color = "#" + rgb_to_hex(rgb)
        char_text_formatting = "<span style=\"background-color: {}\">{}</span>".format(color, generated_text[i])

        f.write(char_text_formatting)
