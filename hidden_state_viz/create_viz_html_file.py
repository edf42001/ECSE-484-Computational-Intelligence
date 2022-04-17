import numpy as np


def rgb_to_hex(rgb):
    # Converts a tuple to a hex string, for use by css
    return '%02x%02x%02x' % rgb


data_file = "hidden_state_viz_data/hidden_data.csv"
text_file = "hidden_state_viz_data/generated_text.txt"
viz_file = "hidden_state.html"

with open(text_file, 'r') as f:
    generated_text = f.read()

# Read the activations, squash between -1 and 1 and multiply by 0.6 so the colors aren't too dark
data = np.loadtxt(data_file, delimiter=",")
data = 0.6 * np.tanh(data)

print("Text: " + generated_text)
length = len(generated_text)

# Monospaced font header
html_string = "<head><style type = \"text/css\">div {font-family: monospace}</style></head>\n"

for node_index in range(data.shape[1]):
    activations_colored_text = ""
    for i in range(length):
        # When we record the data, we feed a character in, and record the output char and node activations
        # But the node activations indicate the current state, which is dependent on the char we feed in
        # Imagine we feed in a ", and the network is now in the quote state, the quote should be colored
        # Anyway, that's why there is a plus 1 here
        node_activation = data[min(i+1, length-1), node_index]

        # Convert node activations less than one to blue, above to red
        if node_activation < 0:
            rgb = (int(255 + 255 * node_activation), int(255 + 255 * node_activation), int(255))
        else:
            rgb = (int(255), int(255 - 255 * node_activation), int(255 - 255 * node_activation))

        color = "#" + rgb_to_hex(rgb)

        # Hardcoded cases to deal with annoying html things such as spaces at start of line not showing up
        if generated_text[i] == "\n":
            char_text = "<br>"
        elif generated_text[i] == " ":
            char_text = "<span style=\"background-color: {}\">{}</span>".format(color, "&nbsp")
        else:
            char_text = "<span style=\"background-color: {}\">{}</span>".format(color, generated_text[i])

        activations_colored_text += char_text

    # Wrap each set of colored text in a div with a horizontal line in between
    node_div_text = "<div id={}>{}</div><hr>\n".format(node_index, activations_colored_text)
    html_string += node_div_text

# Save to the html file
with open(viz_file, "w") as f:
    f.write(html_string)
