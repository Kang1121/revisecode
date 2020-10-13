import numpy as np


def im2c(im, w2c, color):

    w2c = w2c['w2c']
    # print(w2c)
    RR = im[:, :, 2]
    GG = im[:, :, 1]
    BB = im[:, :, 0]

    index_im = 1 + np.floor(RR/8) + 32 * np.floor(GG/8) + 32 * 32 * np.floor(BB/8)
    # print(RR, GG, BB)

    def ab(a, b):
        r = b.shape[0]
        c = b.shape[1]
        for i in range(r):
            for j in range(c):
                # print(b[i][j])
                b[i][j] = a[int(b[i][j])-1]
        return b

    if color == 0:

        # max1 = w2c.max(1)
        w2cM = np.argmax(w2c, 1) + 1
        # print(w2cM[32767])

        index_im = ab(w2cM, index_im)

        out = index_im.reshape(1, -1)

        # print(out.shape)

    return out

    # if color > 0 & color < 12:
    #     w2cM = w2c(:, color)
    #     out = reshape(w2cM(index_im(:)), size(im, 1), size(im, 2))
    #
    #
    # if color == -1:
    #     out = im
    #     [max1, w2cM] = max(w2c, [], 2);
    #     out2 = reshape(w2cM(index_im(:)), size(im, 1), size(im, 2));
    #
    #     for jj=1:size(im, 1):
    #         for ii=1:size(im, 2):
    #         out(jj, ii,:)=color_values{out2(jj, ii)}'*255;
    #
    #
