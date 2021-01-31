from thexp.calculate.schedule import *

if __name__ == '__main__':
    lsc = LinearSchedule(left=0,right=10,start=1,end=0)
    sched = PeriodHalfCosSchedule(left=0, right=100, start=1, end=0)
    from matplotlib import pyplot as plt
    # lsc.plot()
    sched.plot(left=-100,right=300)
    plt.show()
    print(sched)