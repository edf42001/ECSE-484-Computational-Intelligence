import numpy as np


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


data_file = "hidden_state_viz_data/hidden_data.csv"
text_file = "hidden_state_viz_data/generated_text.txt"
viz_file = "hidden_state.html"

with open(text_file, 'r') as f:
    generated_text = f.read()


data = np.loadtxt(data_file, delimiter=",")
data = np.tanh(data)

print("Text: " + generated_text)
length = len(generated_text)

# Notes: Neuron 309 becomes inactive on "units".

html_string = ""
# TODO: investigate node 1300's output is out of range
# 1311 is the parenthase neuron
for node_index in range(data.shape[1]):
    activations_colored_text = ""
    for i in range(length):
        node_activation = data[i, node_index]

        if node_activation < 0:
            rgb = (int(255 + 255 * node_activation), int(255 + 255 * node_activation), int(255))
        else:
            rgb = (int(255), int(255 - 255 * node_activation), int(255 - 255 * node_activation))

        # Debug node_activations->color mapping
        # print(rgb_to_hex(rgb), node_activation)

        color = "#" + rgb_to_hex(rgb)
        # if len(color) > 7:
        #     print(rgb, color, node_activation)

        # Hardcoded cases to deal with annoying html things such as spaces at start of line not showing up
        if generated_text[i] == "\n":
            char_text = "<br>"
        elif generated_text[i] == " ":
            char_text = "<span style=\"background-color: {}\">{}</span>".format(color, "&nbsp")
        else:
            char_text = "<span style=\"background-color: {}\">{}</span>".format(color, generated_text[i])

        activations_colored_text += char_text

    node_div_text = "<div id={}>{}</div>\n".format(node_index, activations_colored_text)
    html_string += node_div_text

# Save to the html file
with open(viz_file, "w") as f:
    f.write(html_string)
