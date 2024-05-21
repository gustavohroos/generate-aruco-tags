import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_aruco_tag(dictionary, id, size, border_size):
    tag = np.ones((size + 2 * border_size, size + 2 * border_size), dtype=np.uint8) * 255
    aruco_marker = cv2.aruco.drawMarker(dictionary, id, size)
    tag[border_size:border_size + size, border_size:border_size + size] = aruco_marker
    return tag

def generate_aruco_pdfs(dictionary, ids, size, border_size):
    for id in ids:
        tag = create_aruco_tag(dictionary, id, size, border_size)

        filename = f"pdf/aruco_tag_{id}.pdf"
        fig = plt.figure(figsize=(11.811, 11.811), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(tag, cmap="gray", aspect='auto')
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

size = 2953
border_size = 591
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
ids = list(range(0, 100))

if __name__ == "__main__":
    generate_aruco_pdfs(dictionary, ids, size, border_size)
