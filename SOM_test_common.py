#######################################
#                                     #
#    SOM Testing Utility Functions    #
#                                     #
#    Author: Lorenzo Mella            #
#                                     #
#######################################


from time import process_time


def test_timer(func, trials=10, with_dry_run=True):
    total_timelapse = 0.
    if with_dry_run:
        func()
    for t in range(trials):
        start = process_time()
        func()
        end = process_time()
        total_timelapse += end - start
    return total_timelapse / trials
