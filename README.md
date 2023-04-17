# DynexSolve
Open Source code for DynexSolve 2.2.5 (reference implementation miner) with redacted Stratum + Mallob communication handlers for network security purposes. Currently supporting CUDA. Please note that this miner is intended mostly for educational purposes. If you want to mine DNX, we recommend using one of the commercial miners which have significantly higher performance and hash-rates.

# Pre-compiled binaries
The official Dynex releases contain pre-compiled binaries of DynexSolve for multiple operating Systems:
https://github.com/dynexcoin/Dynex/releases

# Start mining:

Dynex has developed a proprietary circuit design, the Dynex Neuromorphic Chip, that complements the Dynex ecosystem and turns any modern device into a neuromorphic computing chip that can perform orders of magnitude faster than classical or quantum methodologies for a wide range of applications. Especially due to the dominance of ASICs in the proof-of-work token mining industry, there is a large amount of dormant GPU infrastructure available which can be converted into high performance next-generation neuromorphic computing clusters. All participating nodes together constitute one enormous neuromorphic computing network. Consequently, the platform is capable of performing computations at unprecedented speeds and efficiency – even exceeding quantum computing. 

### Pool mining
Pools are operated by independent pool operators and are there to share mining power and returns. Even if the network difficulty is high, you can still get a share of all blocks mines by a pool. These are usually based on the standard protocol Stratum which is compatible with our DynexSolve miner. Your pool operator can provide you with the right starting configuration / command line options for your environment

When you run DynexSolve for the first time, you will be assigned a unique “Mallob Network ID”. This helps the network to identify and schedule the computational jobs. Mallob is short for “Malleable Load Balancer”, which is a central part of Dynex’ distributed computing management. More information is also avaiable on our website: 
https://dynexcoin.org/get-dnx/#mining

# Build from source
The source coded provided cannot be built without the (redacted) Stratum and Mallob communication handlers. These are not relevant for studiying the functionality of the DynexSolve algorithm and are only made available to selected developers. Please also note that this repository is the reference implementation of the Dynex miner and has limited performance. It is mostly for educational purposes. If you want to mine DNX, we recommend using one of the commercial miners available.

