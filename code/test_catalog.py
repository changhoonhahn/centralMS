'''




'''
import catalog as Cat


if __name__=='__main__': 
    #subhist = Cat.SubhaloHistory(nsnap_ancestor=20)
    #subhist._CheckHistory()
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subhist.Build()
    subhist.Downsample()
