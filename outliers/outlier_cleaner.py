#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    # for index, pred in enumerate(predictions, start=0):
    #     error = abs( pred[0] - net_worths[index][0] )
    #     cleaned_data.append([ages[index][0], net_worths[index][0], error])
    #
    # cleaned_data.sort(key=lambda x: x[2])
    #
    # size = len(cleaned_data)
    # ten_percent = size / 10
    #
    # cleaned_data = cleaned_data[:size-ten_percent]
    errors = (net_worths - predictions)**2
    cleaned_data = zip(ages, net_worths,errors)
    cleaned_data = sorted(cleaned_data, key=lambda x:x[2][0], reverse=True)
    limit = int(len(net_worths) * 0.1)
    
    return cleaned_data[limit:]

