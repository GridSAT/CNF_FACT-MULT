# CNF Generator for Factoring Problems

## Almost all information (c) Paul Purdom and Amr Sabry and taken from [https://cgi.luddy.indiana.edu/~sabry/cnf.html](https://cgi.luddy.indiana.edu/~sabry/cnf.html).

## Authors
- [Paul Purdom](https://legacy.cs.indiana.edu/~pwp/)
- [Amr Sabry](http://www.cs.indiana.edu/~sabry)

## Co-authors
- [GridSAT Stiftung](http://gridsat.io) or IPFS: gridsat.eth/

## News
- **13 Mar 2024:** Input numbers can now automatically be parsed from a simple input.txt of digits separated in lines - CLI input would be none with `n-bit carry-save` per default unless an `input number` and `[n-bit|fast]` `[carry-save|wallace|recursive]` are specified in CLI.
- **12 Mar 2024:** Added support for additional dynamic file output named using the input number provided at runtime, following the format `[number].dimacs` limited to 5 digits and afterwards appended with scientific notation.
- **5 Jul 2014:** Fixed a bug for inputs less than 5.
- **23 Nov 2007:** Fixed a bug that caused the generator to occasionally output clauses with boolean constants instead of variables.
- **1 Jan 2006:** The old CGI servers have been replaced by a new one running a different OS. The code for the simplification has not been recompiled for the new system yet, and hence the simplification of the output does not work now.
- **10 Feb 2005:** Fixed a bug in the simplifier that caused it to sometimes drop some clauses.

## New Functionality in `cnf3.hs`

Building on the capabilities of its predecessors, `cnf3.hs` introduces a batch processing feature:

- **Batch Processing**: `cnf3.hs` processes multiple numbers from input.txt, simplifying large dataset analysis.

To utilize this new feature, simply run the `cnf3.hs` script with your desired parameters. The CNF predicate will be immediately displayed on your screen, and a copy or copies will be saved to a `.dimacs` file in the same location as the script.

## New Functionality in `cnf2.hs`

Based on `cnf1.hs`, `cnf2.hs` introduces an enhanced feature that simultaneously displays the generated Conjunctive Normal Form (CNF) predicate on the screen and saves it to a file. This dual-output functionality ensures immediate view of the results as well as generation process and a permanent record without the need to redirect output manually. Here are the key aspects of this new feature:

- **Simultaneous Output**: Upon execution, `cnf2.hs` prints the CNF predicate to the console, allowing for quick verification and review. Simultaneously, the same output is saved to a file in the current working directory, ensuring that users have easy access to the results for future reference or analysis.

- **Dynamic File Naming**: Outputs are named dynamically using scientific notation in the format `[number].dimacs` or `[prefix]e[exponent].dimacs`.

- **Dual Output**: Immediate console display and file saving of CNF predicates for easy access and review.

To utilize this new feature, simply run the `cnf2.hs` script with your desired parameters. The CNF predicate will be immediately displayed on your screen, and a copy will be saved to a `.dimacs` file in the same location as the script.

## Run Locally
This section provides instructions on how to compile and run `cnf1.hs` / `cnf2.hs` / `cnf3.hs` locally on Linux, macOS, and Windows systems. Ensure you have the Glasgow Haskell Compiler (GHC) installed on your system. For detailed instructions on installing GHC, please [visit the official Haskell website](https://www.haskell.org/get-started/).

### Linux and macOS

1. **Open a terminal**.
2. **Navigate** to the directory containing `cnf1.hs`, `cnf2.hs` or `cnf3.hs` respectively.
3. **Compile** the Haskell script using GHC:

```ghc cnf1.hs -o cnf1``` # Compiles `cnf1.hs` to the executable ```/cnf1```  
and/or  
```ghc cnf2.hs -o cnf2``` # Compiles `cnf2.hs` to the executable ```/cnf2```  
and/or  
```ghc cnf3.hs -o cnf3``` # Compiles `cnf3.hs` to the executable ```/cnf3``` +++ recommended +++  

This command compile `cnf1.hs` / `cnf2.hs` into executables named `cnf1` / `cnf2` respectively.  

4. **Run** the executable with the required arguments:

`/cnf1` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
and/or  
`/cnf2` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
and/or  
`/cnf3` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
or  
`/cnf3` # Batch mode, processes numbers from `input.txt` +++ requires `input.txt` in script directory with min. one digit +++

Replace placeholders as necessary:

`[number]`: The product number to process.  
`[adder_type]`: Choose between n-bit or fast.  
`[multiplier_type]`: Options include carry-save, wallace, or recursive.  

For `/cnf3` without CLI arguments, `n-bit` `carry-save` is default method and input.txt must be in script directory with one digit per line, e.g.:  

1  
22  
333  

### Windows

1. **Open Command Prompt or PowerShell**.
2. **Navigate** to the folder containing `cnf1.hs`, `cnf2.hs` or `cnf3.hs` respectively.
3. **Compile** the Haskell script using GHC:

```ghc cnf1.hs -o cnf1.exe``` # Compiles `cnf1.hs` to the executable `cnf1.exe`  
and/or  
```ghc cnf2.hs -o cnf2.exe``` # Compiles `cnf2.hs` to the executable `cnf2.exe`  
and/or  
```ghc cnf3.hs -o cnf3.exe``` # Compiles `cnf3.hs` to the executable `cnf3.exe` +++ recommended +++  

4. **Run** the executable with the required arguments:

`cnf1.exe` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
and/or  
`cnf2.exe` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
and/or  
`cnf3.exe` `[number]` `[n-bit|fast]` `[carry-save|wallace|recursive]` # Single number processing  
or  
`cnf3.exe` # Batch mode, processes numbers from `input.txt` +++ requires `input.txt` in script directory with min. one digit +++  

Replace placeholders as necessary:

`[number]`: The product number to process.  
`[adder_type]`: Choose between n-bit or fast.  
`[multiplier_type]`: Options include carry-save, wallace, or recursive.  

For `cnf3.exe` without CLI arguments, `n-bit` `carry-save` is default method and input.txt must be in script directory with one digit per line, e.g.:  

1  
22  
333  

### Note
- For any missing Haskell packages or libraries, you may need to install them using `cabal install [package-name]` or `stack install [package-name]`.

## Web Interface
This program generates a Conjunctive Normal Form (CNF) predicate associated with the integer and the circuit that you specify. The predicate will be satisfiable if the integer is composite and it will be unsatisfiable if the integer is prime. The predicate is in the [DIMACS format](http://www.satlib.org/Benchmarks/SAT/satformat.ps) used in [SAT competitions](http://www.satcompetition.org).

## Documentation
For those who want more information, the full source code for the predicate generation is collected in [cnf1.hs](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/cnf1.hs), [cnf2.hs](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/cnf2.hs), [cnf3.hs](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/cnf3.hs). The source code for the simplifier is in this [WEB file](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/Simplify.WEB). The program produces one or two web pages as output depending on whether you select "Simplify output" or not.

The web page for the output predicate begins with a lot of comments. If you are interested, you can use these comments to determine the meaning of each variable in the predicate, and you can discard them if you are not interested. Some of the variables give the bits of the product. Those variables will appear in unit clauses that force them to take on the appropriate value so that they represent the number you input. Some of the variables give the bits of the first factor. Some of the variables give the bits of the second factor. The remaining variables give the values on the wires in the multiplication circuit that is simulated by the predicate. All of the details should be clear from reading the comments. Likewise, the details of how the circuit is wired together should be clear.

To ensure that no factors are found for prime numbers, the circuit requires, for **`n`**-bit inputs, that the first factor have no more than **`(n-1)`** bits and that the second factor has no more than **`n/2`** (rounded up) bits.

The two basic types of multiplication circuits that are implemented are:
- The carry-save multiplier
- [A Suggestion for a Fast Multiplier (IEEE Trans. on Electronic Comp., 1964)](https://ieeexplore.ieee.org/document/4038071) with [the full paper here](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/docs/a_suggestion_for_a_fast_multiplier.pdf).

If you select recursive, then the circuit is built based on the algorithm of Karatsuba (as taken from [The Analysis of Algorithms](https://github.com/GridSAT/CNF_FACT-MULT/blob/main/docs/The_Analysis_of_Algorithms.pdf) by [Paul Purdom](https://legacy.cs.indiana.edu/~pwp/) and [Cynthia A. Brown](https://dblp.org/pid/23/3171.html). When the number of bits in each input becomes less than 20 bits, the recursive multiplier reverts to the Wallace tree multiplier. (The recursive option is only recommended for very large problems.)

For the two basic types of multipliers, the circuit begins by forming all the products **a<sub>i</sub> \* b<sub>j</sub>**  where **a<sub>i</sub>** is a digit from the first number factor and **b<sub>j</sub>** is a digit from the second factor. These products are then added, but each multiplier circuit differs in the details of how they carry out the addition.

In the carry-save circuit, row **`i`** of the circuit adds the product from row **`i`** **(a<sub>i</sub> \* b<sub>\*</sub>)** with the sum and carry (shifted one column) to obtain a new sum and a new carry. A

In the Wallace-tree circuit, the rows (with appropriate shifts) are added in groups of three to produce sums and carries. The sums and carries (with appropriate shifts) are again added in groups of three, and this is repeated until there is just one sum and one carry. Then a special adder is used to add the final sum and the final carry. The products from any row need go through only a logarithmic number of adders before they get to the special adder.

The adder type specifies the special adder. The **`n`**-bit adder uses the algorithm (adapted to binary numbers) taught in grade school. It is simple, but carries must propagate through a linear number of addition stages.

The fast adder is a **`log`**-time adder taken from [Foundations of Computer Science](http://infolab.stanford.edu/~ullman/focs.html) by [Al Aho](http://www1.cs.columbia.edu/~aho/) and [Jeff Ullman](http://infolab.stanford.edu/~ullman/). If you select *Fast* for adder type and *Wallace* for multiplier type, then no path in the circuit has length longer than a small constant times the number of bits needed to specify the input integer. It is an interesting experimental question as to whether such SAT problems are easier.

The predicate generation is written using a [Haskell](https://www.haskell.org) program (inspired by the [Lava library](https://dl.acm.org/doi/pdf/10.1145/289423.289440)). The idea is that one writes completely standard executable specifications for the circuits, and then exploits Haskell type classes to get a non-standard interpretation of the primitives which generates CNF clauses. For example, a half-adder is specified as follows:

```haskell
halfAdd :: Circuit m => (Bit,Bit) -> m (Bit,Bit)
halfAdd (a,b) =
	do carryOut <- and2 (a,b)
		sum <- xor2 (a,b)
		return (carryOut,sum)
```

In the standard interpretation, and2 is defined as the usual operation on booleans:

```haskell
	and2 (Bool x, Bool y) = return (Bool (x && y))
```

In the symbolic interpretation, `and2` instead generates CNF clauses relating its input variables to its output variables:

```haskell
	and2 (b1, b2) =
		do v <- newBitVar
			let or1 = Or [b1, b2, notBit v]
				or2 = Or [b1, notBit b2, notBit v]
				or3 = Or [notBit b1, b2, notBit v]
				or4 = Or [notBit b1, notBit b2, v]
			addDesc [or1,or2,or3,or4]
			return v
```

This approach to generating the predicates is appealing as it enables us to use the full power of Haskell to write circuits naturally, test the circuits extensively using the standard interpretation, and then simply turn a switch on to use the symbolic interpretation. The initial output has many trivial features, including unit clauses. The simplifier applies rules to simplify the predicate without changing the set of solutions. The simplifier has fixed size limits, so it may not work for large problems. (It can handle a maximum of 15,000 variables and 63,000 clauses.)

The simplifier repeatedly applies the following rules to the CNF predicate:

- **Remove subsumed clauses.** If the literals in clause `A` are a subset of those in clause `B`, then remove clause `B`.
- **Do resolutions where at least one parent is subsumed.** If clause `A` is `x l1 ... ln` and clause `B` is `-x l1 ... ln y1 ... yn`, then replace clause `B` with `l1 ... ln y1 ... yn`.
- **Substitute for equalities.** If clause `A` is `l1 -l2` and clause `B` is `-l1 l2`, then replace all occurrences of `l2` with `l1` except in clauses `A` and `B`.


## Applications
- [ATPG template class library](https://sourceforge.net/projects/atpg/) (Library for Combinational/Sequential logic simulation, and fault simulation functionality (semi-PROOFS)).

##
**Last modified**: 03/13/2024 2:36:10 CET

For more information, contact [sabry@cs.indiana.edu](mailto:sabry@cs.indiana.edu) or [GridSAT Stiftung](https://keybase.io/gridsat)
