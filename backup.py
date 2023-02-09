else:
        if speakers[winner1]=='rawan':
            plt.title("Rawan")
            plt.scatter(1, [-18.854410723999997],marker='P')
            ax.plot([-18,-18,-18],'--r')      # upper limit
            ax.plot([-24.5,-24.5,-24.5],'--r')    # lower limit
            plt.scatter(np.arange(0,3), input_mean,marker='P',color='k')
            plt.legend(["Max", "Min","Mean","Input"], loc ="lower right")
            plt.show()
        elif speakers[winner1]=='sara':
            plt.title("Sara")
            plt.scatter(1, [-22.777941949],marker='P')
            ax.plot([-19,-19,-19],'--r')      # upper limit
            ax.plot([-28,-28,-28],'--r')    # lower limit
            plt.scatter(np.arange(0,3), input_mean,marker='P',color='k')
            plt.legend(["Max", "Min","Mean","Input"], loc ="lower right")
            plt.show()
        elif speakers[winner1]=='ammar':
            plt.title("Amar")
            plt.scatter(1, [-45.07298449714],marker='P')
            ax.plot([-44.6,-44.6,-44.6],'--r')      # upper limit
            ax.plot([-45.9,-45.9,-45.9],'--r')    # lower limit
            plt.scatter(np.arange(0,3), input_mean,marker='P',color='k')
            plt.legend(["Max", "Min","Mean","Input"], loc ="lower right")
            plt.show()
    