
X = "../test.out"


# read the text file
def read_the_image_file():
    matr = []
    f = open(X, "r").read().split("\n")[:-1]
    for number in f:
        line = list(number)
        matr.append(line)
    t = [[matr[j][i] for j in range(len(matr))] for i in range(len(matr[0]))]
    return t


def size_of_lungs(img):
    # img = read_the_image_file()
    r = 0  # right (in actual body) lung
    l = 0
    s = 0  # starting point
    m = 0  # middle_separation point
    for line in range(len(img)):
        if img[line][:].count("1") > 0:
            s = line
            break
        else:
            continue
    for line in range(s, len(img)):
        if img[line][:].count("1") == 0:
            m = line
            break
        else:
            s_r = img[line][:].count("1")
            r = r + s_r
    for line in range(m, len(img)):
        s_l = img[line][:].count("1")
        l = l + s_l

    fraction = round(l/r, 3)
    # print("Right lung: %s" % r, "\nLeft lung size: %s" % l, "\nFraction between right and left lung: %s" % fraction)

    return r, l, fraction


size_of_lungs()
