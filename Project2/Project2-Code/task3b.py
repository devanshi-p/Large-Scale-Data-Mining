import helper as hlp
import task1 as t1
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def getBestR(tfidf_matrix,num):
    rank_list = [1,2,3,5,10,20,50,100,300]

    homo_list_svd = []
    comp_list_svd = []
    vscore_list_svd = []
    adjscore_list_svd = []
    infoscore_list_svd = []

    homo_list_nmf = []
    comp_list_nmf = []
    vscore_list_nmf = []
    adjscore_list_nmf = []
    infoscore_list_nmf = []

    for r in rank_list:
        print "................. For r=",r,"......................\n"
        svd_matrix = hlp.getSVD(r)
        svd = svd_matrix.fit_transform(tfidf_matrix)

        nmf = hlp.getNMF(r)
        nmf_matrix = nmf.fit_transform(tfidf_matrix)

        km = hlp.getKmeans(num)

        print "*******With LSI********"
        km.fit(svd)
        homo, comp, vscore, adjscore, infoscore = hlp.getStats(labels,km.labels_)
        homo_list_svd.append(homo)
        comp_list_svd.append(comp)
        vscore_list_svd.append(vscore)
        adjscore_list_svd.append(adjscore)
        infoscore_list_svd.append(infoscore)
        print ""

        print "*******With NMF********"
        km.fit(nmf_matrix)
        homo, comp, vscore, adjscore, infoscore = hlp.getStats(labels,km.labels_)
        homo_list_nmf.append(homo)
        comp_list_nmf.append(comp)
        vscore_list_nmf.append(vscore)
        adjscore_list_nmf.append(adjscore)
        infoscore_list_nmf.append(infoscore)


    print "*******With LSI********"

    plt.plot(rank_list, homo_list_svd)
    plt.ylabel('Homogeneity Score')
    plt.show()

    plt.plot(rank_list, comp_list_svd)
    plt.ylabel('Completeness Score')
    plt.show()

    plt.plot(rank_list, vscore_list_svd)
    plt.ylabel('V-measure Score')
    plt.show()

    plt.plot(rank_list, adjscore_list_svd)
    plt.ylabel('Adjusted rand Score')
    plt.show()

    plt.plot(rank_list, infoscore_list_svd)
    plt.ylabel('Adjusted Mutual Info Score')
    plt.show()

    print "*******With NMF********"

    plt.plot(rank_list, homo_list_nmf)
    plt.ylabel('Homogeneity Score')
    plt.show()

    plt.plot(rank_list, comp_list_nmf)
    plt.ylabel('Completeness Score')
    plt.show()

    plt.plot(rank_list, vscore_list_nmf)
    plt.ylabel('V-measure Score')
    plt.show()

    plt.plot(rank_list, adjscore_list_nmf)
    plt.ylabel('Adjusted rand Score')
    plt.show()

    plt.plot(rank_list, infoscore_list_nmf)
    plt.ylabel('Adjusted Mutual Info Score')
    plt.show()
    
dataset = hlp.fetch_data()
hlp.classify_into_two_class(dataset)
labels = hlp.fetch_labels(dataset)
tfidf_matrix = t1.getTFIDF_matrix(dataset,3)
getBestR(tfidf_matrix,2)