
## Single feature linear regression
To predict the price of a car based on milage

## Data
The data is given in ```data.csv``` and shows two columns - km (milage) and 
price
```shell
km,     price
240000, 3650
139800, 3800
150500, 4400
...
```

## Fitting a line
First is to check if data is having a linear pattern. Corellation vs linear 
regression or trend line. To check if correlation is statistically significant, 
t-score and p-value are also checked. The data is splited for cross validation.

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
Yeah, Docker, I know, it is an overkill, but it's a fun excersize. Plus, I can 
run any version of Python without headaches.

## Install Docker
To install Docker, follow [official installation guides](https://docs.docker.com/get-docker/) 
for your system.

## Building Docker Image
Here ```-t``` or ```--tag``` is used to assign ```name:tag``` to the image
```shell
docker build -t lr .
```

## Running Docker Image
To run the image with the name ```lr``` and use shell terminal by specifying it 
after interactive terminal ```-it``` flag
```shell
docker run -it lr sh
```
And if changes in the code were made, to restart a container run
```shell
docker restart lr
```

## Other Docker commands
To view active containers, use command ```docker ps```:
```shell
docker ps
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS          PORTS                                      NAMES
e9d8c7f5107f   phpmyadmin/phpmyadmin   "/docker-entrypoint.…"   34 seconds ago   Up 32 seconds   0.0.0.0:8001->80/tcp                       php_docker_app-phpmyadmin-1
ffe68784d349   mysql:latest            "docker-entrypoint.s…"   34 seconds ago   Up 32 seconds   3306/tcp, 33060/tcp                        php_docker_app-db-1
3396a1c4b8e3   extended-php:1.0        "docker-php-entrypoi…"   34 seconds ago   Up 32 seconds   0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp   php_docker_app-www-1
```

## Enough is enough
To clean everyting run ```docker system prune -a```
