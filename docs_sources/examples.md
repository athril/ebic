# Examples
The only parameter required in order to run EvoBic is input file.
All other parameters are optional and could be modified by overriding a specific parameter.

To see available options:
```Shell
./evobic -h
```


Run EvoBic for 10 iterations and return 5 biclusters only:
```Shell
./evobic -i input.txt -n 10 -b 5
```

Do not allow negative trends:
```Shell
./evobic -i input.txt -m 0
```

Do not allow approximate trends:
```Shell
./evobic -i input.txt -a 1
```
