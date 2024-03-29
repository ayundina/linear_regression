
## Single/multiple feature linear regression
To predict the price of a car based on milage

## Single feature data
The data is given in ```data.csv``` and shows two columns - km (milage) and 
price
```shell
km,     price
240000, 3650
139800, 3800
150500, 4400
...
```
#### Visualisation
Km is a single feature in the dataset. Shown on x axis. Price is a target value and shown on y axis
![data.csv](https://github.com/ayundina/linear_regression/blob/main/visualisation/data_csv_0.png)

Trained model fits the line in the data cloud
![data.csv-trained](https://github.com/ayundina/linear_regression/blob/main/visualisation/data_csv_trained_0.png)

## How good is the prediction?
R squared value is used to calculate how well the regression line represents 
the data. The greater the R squared value, the better the fit.
$$r^2 = 1 - (SSR/SST)$$

Where SSR is a sum of squared residuals — sum of a diff between the regression 
line and each indivudual data point.
$$SSR = \sum(y-\hat{y}3)$$

And SST is a sum of squares total — sum of a diff between individual data point 
and their arithmetic mean.
$$SST = \sum(y-\bar{y})$$

## Docker?
An overkill, but it's a fun excersize. Plus, I can run any version of Python 
without headaches.

#### Install Docker
To install Docker, follow [official installation guides](https://docs.docker.com/get-docker/) 
for your system.

#### First: Building Docker Image
Here ```-t``` or ```--tag``` is used to assign ```name:tag``` to the image
```shell
docker build -t lr .
```

#### Second: Running Docker Image
To run the image with the name ```lr``` and use shell terminal by specifying it 
after interactive terminal ```-it``` flag
```shell
docker run -it lr sh
```
Or if changes in the code were made, to rebuild and rerun the image
```shell
docker run -it $(docker build -q .)
```
Where command inside ```$()``` is executed first and the output is passed to the
 initial command.
Flag ```-q``` is used to silence build output.

#### Other Docker commands
To view images, run ```docker image ls```
To view containers, run ```docker container ls```
To view active containers, use command ```docker ps -a```

#### Last: Enough is enough
Then, to clean everyting run ```docker system prune -a -f```
```-a``` for dangling and unused images
```-f``` for auto yes

#### Docker docks. Just in case
Check out [docker documentation](https://docs.docker.com/reference/)

## Makefile?
“Isn’t Make an old tool?” or “Isn’t it only meant for C and C++ projects?”. 
In reality, this couldn’t be farther from the truth. It’s correct that Make 
is a utility developed back in the ’70s and ’80s, and yes, it’s perceived as 
being tied to C and C++ applications, but that doesn’t mean it doesn’t have 
its advantages in other projects.

To build, run and then clean an image, run ```make all``` 