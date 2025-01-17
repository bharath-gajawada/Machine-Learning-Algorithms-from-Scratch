#6.1, 6.3

import numpy as np
import pandas as pd

import sys
sys.path.append("../..")

import models.k_means.k_means as Km
import models.pca.pca as PCA
import models.gmm.gmm as GMM

def main():

    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    pca = PCA.PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    #6.1

    km = Km.KMeans(3) # k2
    km.fit(X_pca)

    labels = km.predict(X_pca)
    cost = km.getCost(X_pca)
    print(f"Labels: {labels}")
    print(f"Cost: {cost}")

    # Labels: [0 1 2 2 2 1 0 0 1 0 2 1 2 0 0 0 0 0 0 0 1 2 0 0 0 0 0 1 1 0 0 0 2 0 2 1 1
    #  0 0 0 1 0 1 1 2 0 1 0 0 0 0 0 0 1 0 0 2 1 2 1 2 2 2 1 0 1 0 2 1 0 2 1 1 1
    #  0 2 0 0 0 0 0 0 0 2 0 0 1 2 0 0 0 1 1 2 0 0 1 0 2 1 0 0 0 0 1 2 0 0 0 0 1
    #  1 0 1 0 2 1 0 0 0 0 0 0 0 2 1 0 1 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0
    #  0 0 0 0 0 1 2 0 1 0 0 2 0 2 0 0 0 2 0 0 1 2 0 1 2 1 2 0 0 0 1 0 0 0 0 0 0
    #  1 0 2 0 1 2 0 0 0 0 0 0 0 0 0]
    # Cost: 226.25653407343304


    
    
    #6.3

    gmm = GMM.GMM(n_components=3) #k2
    gmm.fit(X_pca)

    params = gmm.get_params()
    print("Means:\n", params['centers'])
    print("Covariances:\n", params['covariances'])
    print("Weights:\n", params['weights'])
    membership = gmm.get_membership()
    print("Memebership:\n", membership)
    likelihood = gmm.get_likelihood(X_pca)
    print("Likelihood:\n", likelihood)

    # Means:
    #  [[ 0.39828876  0.34150207]
    #  [-2.01559498  0.18403418]
    #  [ 0.90286838 -1.71529088]]
    # Covariances:
    #  [[[0.8077693  0.1459261 ]
    #   [0.1459261  0.43833322]]

    #  [[1.07204588 0.22730785]
    #   [0.22730785 0.13564942]]

    #  [[2.07357957 0.51732591]
    #   [0.51732591 0.55601885]]]
    # Weights:
    #  [0.65245204 0.19656042 0.15098754]
    # Memebership:
    #  [[7.23986497e-01 2.75996002e-01 1.75006464e-05]
    #  [2.83552191e-01 7.16441668e-01 6.14124463e-06]
    #  [7.10108286e-01 5.47048330e-08 2.89891659e-01]
    #  [5.22699682e-05 1.08055241e-31 9.99947730e-01]
    #  [2.24184852e-03 9.98028198e-23 9.97758151e-01]
    #  [4.30355844e-04 9.99569398e-01 2.46002266e-07]
    #  [9.91468244e-01 3.39171282e-03 5.14004352e-03]
    #  [5.76093402e-01 4.23870420e-01 3.61786651e-05]
    #  [4.64505934e-03 9.95353609e-01 1.33203697e-06]
    #  [9.46286330e-01 8.40126242e-06 5.37052690e-02]
    #  [1.13550448e-01 1.06454109e-11 8.86449552e-01]
    #  [1.91978756e-01 8.08019440e-01 1.80413895e-06]
    #  [2.28497122e-05 1.49388012e-36 9.99977150e-01]
    #  [9.92342130e-01 7.65018081e-03 7.68914037e-06]
    #  [9.96256940e-01 9.85197023e-04 2.75786263e-03]
    #  [9.92715105e-01 5.29512171e-03 1.98977295e-03]
    #  [9.89805223e-01 9.83274727e-03 3.62029327e-04]
    #  [6.70195446e-01 3.29711776e-01 9.27782698e-05]
    #  [7.43721220e-01 2.56254886e-01 2.38933378e-05]
    #  [9.74824345e-01 2.49638186e-02 2.11836090e-04]
    #  [9.74913159e-01 1.65619421e-02 8.52489868e-03]
    #  [8.85575529e-01 3.59289450e-06 1.14420878e-01]
    #  [9.94672379e-01 2.08483478e-03 3.24278619e-03]
    #  [7.25091262e-01 2.74897502e-01 1.12369000e-05]
    #  [9.93799620e-01 4.16255630e-03 2.03782374e-03]
    #  [9.94242060e-01 3.65506222e-03 2.10287761e-03]
    #  [9.64810629e-01 4.96408668e-05 3.51397297e-02]
    #  [1.06553483e-01 8.93441903e-01 4.61405453e-06]
    #  [1.20895603e-03 9.98790791e-01 2.52475625e-07]
    #  [9.94048689e-01 2.75330948e-03 3.19800199e-03]
    #  [9.95410768e-01 1.38676429e-03 3.20246802e-03]
    #  [9.94155737e-01 5.84306628e-03 1.19668616e-06]
    #  [7.15481409e-01 4.74813036e-08 2.84518543e-01]
    #  [9.97351604e-01 2.63938400e-03 9.01208993e-06]
    #  [1.13232288e-03 2.66514512e-21 9.98867677e-01]
    #  [3.93257741e-01 6.06687661e-01 5.45982917e-05]
    #  [1.70001549e-03 9.98274814e-01 2.51700289e-05]
    #  [9.97571622e-01 2.42503334e-03 3.34461169e-06]
    #  [7.25018785e-01 2.74892857e-01 8.83579651e-05]
    #  [9.26374925e-01 4.02534414e-06 7.36210495e-02]
    #  [4.62614275e-03 9.95373038e-01 8.18987288e-07]
    #  [9.65651851e-01 3.39219495e-02 4.26199217e-04]
    #  [4.33464444e-02 9.56652612e-01 9.43492526e-07]
    #  [9.13239721e-01 6.79001147e-05 8.66923792e-02]
    #  [8.21988072e-03 1.67558452e-19 9.91780119e-01]
    #  [9.93409198e-01 3.24303697e-03 3.34776527e-03]
    #  [9.77024150e-01 1.67896791e-02 6.18617110e-03]
    #  [8.78556278e-01 1.21301629e-01 1.42092177e-04]
    #  [9.68606273e-01 2.96235347e-02 1.77019209e-03]
    #  [8.30386466e-01 1.69606995e-01 6.53922939e-06]
    #  [9.73393274e-01 8.52765749e-05 2.65214494e-02]
    #  [8.17383948e-01 1.82322652e-01 2.93400324e-04]
    #  [9.22501432e-01 7.74931884e-02 5.37946144e-06]
    #  [9.64108319e-01 2.59303061e-02 9.96137479e-03]
    #  [9.90624787e-01 3.98470826e-04 8.97674262e-03]
    #  [9.96665647e-01 1.27949155e-03 2.05486187e-03]
    #  [1.71865249e-01 2.56647228e-12 8.28134751e-01]
    #  [3.94425376e-01 6.05567577e-01 7.04704462e-06]
    #  [6.12756480e-05 6.40008021e-33 9.99938724e-01]
    #  [1.21176022e-04 9.99878328e-01 4.96097835e-07]
    #  [5.49294243e-02 6.72648362e-14 9.45070576e-01]
    #  [2.00716840e-01 6.15048894e-12 7.99283160e-01]
    #  [1.38966151e-01 1.63650277e-11 8.61033849e-01]
    #  [4.04373527e-01 5.95551896e-01 7.45767912e-05]
    #  [9.87097260e-01 4.75317379e-04 1.24274228e-02]
    #  [1.05234794e-01 8.94653417e-01 1.11788782e-04]
    #  [9.87951603e-01 1.19832541e-02 6.51429205e-05]
    #  [5.77703953e-03 2.12767101e-18 9.94222960e-01]
    #  [1.11318925e-03 9.98699280e-01 1.87530776e-04]
    #  [9.75523999e-01 2.44752676e-02 7.33826914e-07]
    #  [5.05125174e-04 2.84556192e-24 9.99494875e-01]
    #  [7.29052593e-01 6.91964702e-06 2.70940487e-01]
    #  [9.59881488e-01 4.00461630e-04 3.97180504e-02]
    #  [1.02090383e-01 8.97811479e-01 9.81374779e-05]
    #  [9.90441467e-01 9.41107262e-03 1.47460041e-04]
    #  [3.81540879e-03 2.00300881e-21 9.96184591e-01]
    #  [8.17536483e-01 9.25540745e-07 1.82462592e-01]
    #  [9.35107748e-01 6.46411294e-02 2.51122198e-04]
    #  [9.92649679e-01 3.60532415e-04 6.98978849e-03]
    #  [8.40055075e-01 1.59848632e-01 9.62932851e-05]
    #  [5.87408270e-01 4.12572579e-01 1.91511761e-05]
    #  [9.81724519e-01 7.73575407e-05 1.81981239e-02]
    #  [9.78307297e-01 5.94391861e-05 2.16332639e-02]
    #  [1.22656289e-03 4.16468321e-24 9.98773437e-01]
    #  [9.61649338e-01 3.82641995e-02 8.64626807e-05]
    #  [9.77017177e-01 9.34207847e-04 2.20486147e-02]
    #  [1.83120022e-02 9.81672099e-01 1.58990113e-05]
    #  [6.32021978e-01 1.20332889e-08 3.67978010e-01]
    #  [9.93161750e-01 1.06756167e-03 5.77068811e-03]
    #  [9.91619384e-01 2.88848281e-04 8.09176812e-03]
    #  [8.57398634e-01 1.42596393e-01 4.97352677e-06]
    #  [2.05725631e-05 1.86003579e-25 9.99979427e-01]
    #  [5.48154384e-01 4.51733812e-01 1.11804563e-04]
    #  [5.23186414e-03 8.35473510e-15 9.94768136e-01]
    #  [9.83621261e-01 1.28517324e-02 3.52700658e-03]
    #  [9.96534430e-01 2.27759289e-03 1.18797691e-03]
    #  [2.32138094e-01 7.67846722e-01 1.51842101e-05]
    #  [9.96763405e-01 3.21487679e-03 2.17181223e-05]
    #  [6.32331158e-01 1.20948758e-08 3.67668830e-01]
    #  [3.31845388e-01 6.68099577e-01 5.50345379e-05]
    #  [9.74301254e-01 2.55854416e-02 1.13304358e-04]
    #  [9.60737912e-01 1.69486359e-05 3.92451392e-02]
    #  [9.84485565e-01 1.75982083e-04 1.53384526e-02]
    #  [9.81676286e-01 1.77063265e-02 6.17387402e-04]
    #  [5.26177345e-02 9.47376701e-01 5.56432000e-06]
    #  [5.90637033e-01 1.48169766e-08 4.09362952e-01]
    #  [9.99804312e-01 1.91587924e-04 4.10023765e-06]
    #  [9.31349649e-01 5.17606185e-06 6.86451748e-02]
    #  [9.84824993e-01 1.43935016e-02 7.81505442e-04]
    #  [9.89320097e-01 1.03063408e-02 3.73562007e-04]
    #  [3.17415364e-03 9.96823111e-01 2.73563868e-06]
    #  [7.84804818e-01 2.06657782e-01 8.53740083e-03]
    #  [9.73751415e-01 2.58054841e-02 4.43101179e-04]
    #  [6.42481547e-02 9.35730793e-01 2.10526312e-05]
    #  [9.79189730e-01 2.06038263e-02 2.06444051e-04]
    #  [1.32698232e-02 8.49502166e-18 9.86730177e-01]
    #  [4.75891327e-02 4.07644678e-05 9.52370103e-01]
    #  [9.93126115e-01 6.71907580e-03 1.54809000e-04]
    #  [9.95094204e-01 6.43589428e-04 4.26220679e-03]
    #  [9.89029827e-01 1.91011464e-03 9.06005814e-03]
    #  [8.86846307e-01 1.13110590e-01 4.31031249e-05]
    #  [9.70908401e-01 2.87572164e-02 3.34382945e-04]
    #  [9.87065741e-01 4.25089162e-04 1.25091695e-02]
    #  [9.81868967e-01 1.76343182e-02 4.96715158e-04]
    #  [1.94757517e-04 3.49389098e-29 9.99805242e-01]
    #  [2.42374948e-03 9.97574241e-01 2.00923501e-06]
    #  [9.90035099e-01 9.49927694e-03 4.65623991e-04]
    #  [4.23734494e-03 9.95752487e-01 1.01679655e-05]
    #  [6.99925091e-03 9.92989798e-01 1.09514819e-05]
    #  [5.54848306e-01 4.43334615e-01 1.81707843e-03]
    #  [8.18877417e-02 9.18042360e-01 6.98980166e-05]
    #  [1.07024120e-03 9.98923344e-01 6.41461526e-06]
    #  [9.74749168e-01 4.26570950e-05 2.52081746e-02]
    #  [4.76099705e-02 9.52366482e-01 2.35476495e-05]
    #  [9.88353513e-01 2.00794029e-04 1.14456930e-02]
    #  [9.25155391e-01 7.48115315e-02 3.30771798e-05]
    #  [8.94632999e-01 1.05338856e-01 2.81449673e-05]
    #  [8.94426976e-01 1.05297402e-01 2.75622092e-04]
    #  [9.98911196e-01 1.05214584e-03 3.66581303e-05]
    #  [8.57709002e-02 9.14208618e-01 2.04816541e-05]
    #  [9.99388841e-01 6.09904963e-04 1.25404682e-06]
    #  [9.68231451e-01 3.08159560e-02 9.52593127e-04]
    #  [8.58718557e-01 1.40847275e-01 4.34167502e-04]
    #  [9.46133150e-01 4.96772343e-02 4.18961524e-03]
    #  [9.29321087e-01 7.06726849e-02 6.22842010e-06]
    #  [9.99649411e-01 3.49961687e-04 6.27239956e-07]
    #  [9.86816119e-01 1.28364715e-02 3.47408988e-04]
    #  [9.74715828e-01 2.52822489e-02 1.92343471e-06]
    #  [9.79709277e-01 1.89318594e-02 1.35886347e-03]
    #  [8.29019773e-01 1.70386405e-01 5.93821856e-04]
    #  [9.66274058e-01 3.37040846e-02 2.18570507e-05]
    #  [9.71398619e-01 3.69327101e-05 2.85644481e-02]
    #  [9.69109848e-01 3.00008470e-02 8.89304604e-04]
    #  [7.44907072e-02 9.25444195e-01 6.50974865e-05]
    #  [9.10508430e-01 3.70095233e-06 8.94878687e-02]
    #  [9.89503637e-01 1.00508729e-02 4.45490524e-04]
    #  [5.21702960e-01 4.78287724e-01 9.31618233e-06]
    #  [8.01389703e-01 1.98436859e-01 1.73438075e-04]
    #  [9.93069698e-01 4.00525455e-04 6.52977613e-03]
    #  [6.94174697e-02 3.57629445e-14 9.30582530e-01]
    #  [9.85658858e-01 7.75442864e-03 6.58671356e-03]
    #  [8.29203319e-03 2.12072267e-19 9.91707967e-01]
    #  [9.91880751e-01 3.86766698e-04 7.73248195e-03]
    #  [9.75664190e-01 4.46463756e-05 2.42911632e-02]
    #  [9.95314361e-01 9.82954639e-04 3.70268434e-03]
    #  [1.33662435e-02 2.97580810e-18 9.86633757e-01]
    #  [9.66824844e-01 3.31710671e-02 4.08914589e-06]
    #  [9.99056473e-01 9.43082557e-04 4.44626250e-07]
    #  [3.68731983e-03 9.96285283e-01 2.73972606e-05]
    #  [5.21180823e-02 8.81305252e-13 9.47881918e-01]
    #  [9.68817014e-01 3.11759870e-02 6.99938827e-06]
    #  [3.92425652e-02 9.60755517e-01 1.91813331e-06]
    #  [1.94711376e-02 5.76991417e-16 9.80528862e-01]
    #  [3.06318518e-04 9.99693232e-01 4.49765243e-07]
    #  [9.46303355e-03 4.66947672e-19 9.90536966e-01]
    #  [9.94141451e-01 4.43958551e-03 1.41896311e-03]
    #  [9.83835182e-01 1.52996523e-02 8.65165942e-04]
    #  [8.88882523e-01 1.11112818e-01 4.65967564e-06]
    #  [2.16216371e-01 7.83772565e-01 1.10648481e-05]
    #  [9.87640400e-01 1.23022772e-02 5.73231376e-05]
    #  [8.43956749e-01 1.55989349e-01 5.39021411e-05]
    #  [9.41562203e-01 5.82328278e-02 2.04969198e-04]
    #  [9.95610859e-01 1.75917319e-03 2.62996775e-03]
    #  [9.96304281e-01 8.86605468e-04 2.80911356e-03]
    #  [9.05890642e-01 9.41030488e-02 6.30912699e-06]
    #  [1.02860668e-01 8.97132259e-01 7.07284304e-06]
    #  [9.86475633e-01 1.28986021e-02 6.25764856e-04]
    #  [4.48871768e-03 5.79495115e-21 9.95511282e-01]
    #  [9.96118308e-01 3.87826507e-03 3.42650210e-06]
    #  [6.80904160e-01 3.00537156e-01 1.85586837e-02]
    #  [4.24092369e-02 1.23672852e-13 9.57590763e-01]
    #  [9.89911573e-01 4.82461878e-03 5.26380842e-03]
    #  [8.48756307e-01 1.51054127e-01 1.89565369e-04]
    #  [9.16005691e-01 8.39129984e-02 8.13109891e-05]
    #  [9.87060809e-01 1.25957732e-02 3.43417442e-04]
    #  [8.68537383e-01 1.30854815e-01 6.07802277e-04]
    #  [9.80959722e-01 1.59409538e-02 3.09932396e-03]
    #  [9.57616698e-01 4.13483845e-02 1.03491742e-03]
    #  [9.95375313e-01 3.82905224e-03 7.95635063e-04]
    #  [9.99955484e-01 4.40728361e-05 4.43412799e-07]]
    # Likelihood:
    #  -2.9194715489902894

if __name__ == "__main__":
    main()

