import statistics as s

t = [28,31,29,28,32,35,28,30,33,31]
w = ["Sunny","Cloudy","Sunny","Rainy","Sunny","Sunny","Cloudy","Sunny","Rainy","Sunny"]

print("Temperature:", t, "\nWeather:", w)
print("Mean:", s.mean(t))
print("Median:", s.median(t))
print("Mode:", s.mode(t))
print("Variance:", s.variance(t))
print("Standard Deviation:", s.stdev(t))
print("Weather Mode:", s.mode(w))