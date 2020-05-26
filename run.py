from src.discriminator import run_discriminator


# The plan:
# + train the generator for 1 step, 
# + take some estimations for a trainingset, 
# + throw in some amount of correct datapoints (really need you here, see first question), 
# + let the discrimator train for 1 step, 
# + and then repeat this thing.
def main():
    run_discriminator()

if __name__ == '__main__':
    main()