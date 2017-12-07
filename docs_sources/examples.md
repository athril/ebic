# Examples
The only parameter required in order to run *ebic* is input file.
All other parameters are optional and could be modified by overriding a specific parameter.

To see available options:
```Shell
./ebic -h
```


Run ebic for 10 iterations and return 5 biclusters only:
```Shell
./ebic -i input.txt -n 10 -b 5
```

Do not allow negative trends:
```Shell
./ebic -i input.txt -m 0
```

Do not allow approximate trends:
```Shell
./ebic -i input.txt -a 1
```
