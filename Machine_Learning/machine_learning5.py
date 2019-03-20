import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time 


date,bid,ask = np.loadtxt('GBPUSD1d.txt', unpack = True, delimiter=',')






    

def perChange(start, end):
    if start != 0.0:
        par = ((float(end) - start)/(abs(start)))*100.00
        return par
    else:
        return 50
        




def pattfinder():
    
    x = len(avgLine) - 60
    y = 31
    start_time = time.time()
    while y < x:
        
        pattern = []
        p1 = perChange(avgLine[y-30], avgLine[y-29])
        p2 = perChange(avgLine[y-30], avgLine[y-28])
        p3 = perChange(avgLine[y-30], avgLine[y-27])
        p4 = perChange(avgLine[y-30], avgLine[y-26])
        p5 = perChange(avgLine[y-30], avgLine[y-25])
        p6 = perChange(avgLine[y-30], avgLine[y-24])
        p7 = perChange(avgLine[y-30], avgLine[y-23])
        p8 = perChange(avgLine[y-30], avgLine[y-22])
        p9 = perChange(avgLine[y-30], avgLine[y-21])
        p10 = perChange(avgLine[y-30], avgLine[y-20])
        p11 = perChange(avgLine[y-30], avgLine[y-19])
        p12 = perChange(avgLine[y-30], avgLine[y-18])
        p13 = perChange(avgLine[y-30], avgLine[y-17])
        p14 = perChange(avgLine[y-30], avgLine[y-16])
        p15 = perChange(avgLine[y-30], avgLine[y-15])
        p16 = perChange(avgLine[y-30], avgLine[y-14])
        p17 = perChange(avgLine[y-30], avgLine[y-13])
        p18 = perChange(avgLine[y-30], avgLine[y-12])
        p19 = perChange(avgLine[y-30], avgLine[y-11])
        p20 = perChange(avgLine[y-30], avgLine[y-10])
        p21 = perChange(avgLine[y-30], avgLine[y-9])
        p22 = perChange(avgLine[y-30], avgLine[y-8])
        p23 = perChange(avgLine[y-30], avgLine[y-7])
        p24 = perChange(avgLine[y-30], avgLine[y-6])
        p25 = perChange(avgLine[y-30], avgLine[y-5])
        p26 = perChange(avgLine[y-30], avgLine[y-4])
        p27 = perChange(avgLine[y-30], avgLine[y-3])
        p28 = perChange(avgLine[y-30], avgLine[y-2])
        p29 = perChange(avgLine[y-30], avgLine[y-1])
        p30 = perChange(avgLine[y-30], avgLine[y])

        outcomeRange = avgLine[y+20:y+30]
        currentpoint  = avgLine[y]
        future_mean = (np.mean(outcomeRange))
        
        future_outcome =  perChange(currentpoint, future_mean)
        

        pattern.append(p1)
        pattern.append(p2)
        pattern.append(p3)
        pattern.append(p4)
        pattern.append(p5)
        pattern.append(p6)
        pattern.append(p7)
        pattern.append(p8)
        pattern.append(p9)
        pattern.append(p10)
        pattern.append(p11)
        pattern.append(p12)
        pattern.append(p13)
        pattern.append(p14)
        pattern.append(p15)
        pattern.append(p16)
        pattern.append(p17)
        pattern.append(p18)
        pattern.append(p19)
        pattern.append(p20)
        
        pattern.append(p21)
        pattern.append(p22)
        pattern.append(p23)
        pattern.append(p24)
        pattern.append(p25)
        pattern.append(p26)
        pattern.append(p27)
        pattern.append(p28)
        pattern.append(p29)
        pattern.append(p30)
        pattA.append(pattern)
        perf.append(future_mean)
        y = y + 1

    end_time = time.time()
       

    print(end_time - start_time)

def currentpat():
    

    cp1 = perChange(avgLine[-31], avgLine[-30])
    cp2 = perChange(avgLine[-31], avgLine[-29])
    cp3 = perChange(avgLine[-31], avgLine[-28])
    cp4 = perChange(avgLine[-31], avgLine[-27])
    cp5 = perChange(avgLine[-31], avgLine[-26])
    cp6 = perChange(avgLine[-31], avgLine[-25])
    cp7 = perChange(avgLine[-31], avgLine[-24])
    cp8 = perChange(avgLine[-31], avgLine[-23])
    cp9 = perChange(avgLine[-31], avgLine[-22])
    cp10 = perChange(avgLine[-31], avgLine[-21])
    cp11 = perChange(avgLine[-31], avgLine[-20])
    cp12 = perChange(avgLine[-31], avgLine[-19])
    cp13 = perChange(avgLine[-31], avgLine[-18])
    cp14 = perChange(avgLine[-31], avgLine[-17])
    cp15 = perChange(avgLine[-31], avgLine[-16])
    cp16 = perChange(avgLine[-31], avgLine[-15])
    cp17 = perChange(avgLine[-31], avgLine[-14])
    cp18 = perChange(avgLine[-31], avgLine[-13])
    cp19 = perChange(avgLine[-31], avgLine[-12])
    cp20 = perChange(avgLine[-31], avgLine[-11])
    
    cp21 = perChange(avgLine[-31], avgLine[-10])
    cp22 = perChange(avgLine[-31], avgLine[-9])
    cp23 = perChange(avgLine[-31], avgLine[-8])
    cp24 = perChange(avgLine[-31], avgLine[-7])
    cp25 = perChange(avgLine[-31], avgLine[-6])
    cp26 = perChange(avgLine[-31], avgLine[-5])
    cp27 = perChange(avgLine[-31], avgLine[-4])
    cp28 = perChange(avgLine[-31], avgLine[-3])
    cp29 = perChange(avgLine[-31], avgLine[-2])
    cp30 = perChange(avgLine[-31], avgLine[-1])
    
    patfor.append(cp1)
    patfor.append(cp2)
    patfor.append(cp3)
    patfor.append(cp4)
    patfor.append(cp5)
    patfor.append(cp6)
    patfor.append(cp7)
    patfor.append(cp8)
    patfor.append(cp9)
    patfor.append(cp10)
    patfor.append(cp11)
    patfor.append(cp12)
    patfor.append(cp13)
    patfor.append(cp14)
    patfor.append(cp15)
    patfor.append(cp16)
    patfor.append(cp17)
    patfor.append(cp18)
    patfor.append(cp19)
    patfor.append(cp20)
    patfor.append(cp21)
    patfor.append(cp22)
    patfor.append(cp23)
    patfor.append(cp24)
    patfor.append(cp25)
    patfor.append(cp26)
    patfor.append(cp27)
    patfor.append(cp28)
    patfor.append(cp29)
    patfor.append(cp30)

    print(patfor)

   

def patterReco():
    for eachPat in pattA:
       # print('the value of patt', eachPat[0], patfor[0])
        #print('the patt is',abs(perChange(eachPat[0], patfor[0])))      
        sim1 = 100.0 - abs(perChange(eachPat[0], patfor[0]))
        sim2 = 100.0 - abs(perChange(eachPat[1], patfor[1]))
        sim3 = 100.0 - abs(perChange(eachPat[2], patfor[2]))
        sim4 = 100.0 - abs(perChange(eachPat[3], patfor[3]))
        sim5 = 100.0 - abs(perChange(eachPat[4], patfor[4]))
        sim6 = 100.0 - abs(perChange(eachPat[5], patfor[5]))
        sim7 = 100.0 - abs(perChange(eachPat[6], patfor[6]))
        sim8 = 100.0 - abs(perChange(eachPat[7], patfor[7]))
        sim9 = 100.0 - abs(perChange(eachPat[8], patfor[8]))

        sim10 = 100.0 - abs(perChange(eachPat[9], patfor[9]))
        sim11 = 100.0 - abs(perChange(eachPat[10], patfor[10]))
        sim12 = 100.0 - abs(perChange(eachPat[11], patfor[11]))
        sim13 = 100.0 - abs(perChange(eachPat[12], patfor[12]))
        sim14 = 100.0 - abs(perChange(eachPat[13], patfor[13]))
        sim15 = 100.0 - abs(perChange(eachPat[14], patfor[14]))
        sim16 = 100.0 - abs(perChange(eachPat[15], patfor[15]))
        sim17 = 100.0 - abs(perChange(eachPat[16], patfor[16]))
        sim18 = 100.0 - abs(perChange(eachPat[17], patfor[17]))
        sim19 = 100.0 - abs(perChange(eachPat[18], patfor[18]))
        sim20 = 100.0 - abs(perChange(eachPat[19], patfor[19]))
        
        sim21 = 100.0 - abs(perChange(eachPat[20], patfor[20]))
        sim22 = 100.0 - abs(perChange(eachPat[21], patfor[21]))
        sim23 = 100.0 - abs(perChange(eachPat[22], patfor[22]))
        sim24 = 100.0 - abs(perChange(eachPat[23], patfor[23]))
        sim25 = 100.0 - abs(perChange(eachPat[24], patfor[24]))
        sim26 = 100.0 - abs(perChange(eachPat[25], patfor[25]))
        sim27 = 100.0 - abs(perChange(eachPat[26], patfor[26]))
        sim28 = 100.0 - abs(perChange(eachPat[27], patfor[27]))
        sim29 = 100.0 - abs(perChange(eachPat[28], patfor[28]))
        sim30 = 100.0 - abs(perChange(eachPat[29], patfor[29]))

        
        howsim = (sim1+sim2+sim3+sim4+sim5+sim6+sim7+sim8+sim9+sim10+
                  sim11+sim12+sim13+sim14+sim15+sim16+sim17+sim18+sim19+sim20+
                  sim21+sim22+sim23+sim24+sim25+sim26+sim27+sim28+sim29+sim30)/30.0
        if howsim > 40:
            patdex = pattA.index(eachPat)

            print ('#####')
           # print (perf)
            print ('#####')
            print (eachPat)
            print ('#####')
            #print (perf[patdex])
pattA = []
perf = []
patfor = []
avgLine = ((bid+ask)/2)



pattfinder()
currentpat()
patterReco()


'''
def graphRawFX():
    
    
    
    fig = plt.figure(figsize =(10,7))
    ax1 = plt.subplot2grid((40,40),(0,0),rowspan=40,colspan=40)

    ax1.plot(sec,bid)

    ax1.plot(sec,ask)
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1_2 = ax1.twinx()
    ax1_2.fill_between(date, 0, (ask-bid), facecolor='g', alpha = .3)
    plt.subplots_adjust(bottom = .23)
    

   
    plt.grid(True)
    plt.show()

'''


















