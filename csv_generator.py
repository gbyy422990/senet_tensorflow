import os
import csv



def create_csv(dirname):
    path = '/notebooks/airbus/data/'+ dirname +'/'
    name = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    with open ('data_'+dirname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for n in name:
            if n[-4:] != '.DS_Store':
                print(n)
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow(['/notebooks/airbus/data/' + str(dirname) + '/' + str(n),
                                 '/notebooks/airbus/data/' + str(dirname) + 'label/' + str(n)])
            else:
                pass

if __name__ == "__main__":
    create_csv('train')
    create_csv('val')
