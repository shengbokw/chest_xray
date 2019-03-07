
X = "../test.out"


# read the text file
def read_the_image_file():
    matr = []
    f = open(X, "r").read().split("\n")[:-1]
    for number in f:
        line = list(map(int, list(number)))
        matr.append(line)
    # t = [[matr[j][i] for j in range(len(matr))] for i in range(len(matr[0]))]
    return matr


def size_of_lungs(img):
    # img = read_the_image_file()
    t = [[img[j][i] for j in range(len(img))] for i in range(len(img[0]))]
    r = 0  # right (in actual body) lung
    l = 0
    s = 0  # starting point
    m = 0  # middle_separation point
    for line in range(len(t)):
        if sum(t[line][:]) > 0:
            s = line
            break
        else:
            continue
    for line in range(s, len(t)):
        if sum(t[line][:]) == 0:
            m = line
            break
        else:
            s_r = sum(t[line][:])
            r = r + s_r
    for line in range(m, len(t)):
        s_l = sum(t[line][:])
        l = l + s_l

    fraction = round(l/r, 3)
    # print("Right lung: %s" % r, "\nLeft lung size: %s" % l, "\nFraction between right and left lung: %s" % fraction)

    return r, l, fraction
