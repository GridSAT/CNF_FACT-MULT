{-# LANGUAGE ScopedTypeVariables, MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies, FlexibleInstances #-}

{--

cnf1.hs 
by Paul Purdom and Amr Sabry 

Original version -- March 19, 2003
Version 1 -- updated November 23, 2007

Sean Weaver reported spurious True/False appearing in output: the multiplexer
was not handling the possibilities of having one of the inputs being
known. Fixed by adding four new clauses to the symbolic execution of mux3.

--}

--------------------------------------------------------------------------------
-- Our goal is to describe circuits naturally and then automatically produce a 
-- formula in conjunctive normal form which is equivalent to the circuit

-- This implementation is inspired by Lava:
-- We have different interpretations for circuits: the standard
-- interpretation simulates the circuit; the symbolic one gives us a
-- textual representation; the CNF one gives us an equivalent formula
-- in CNF. The representations are abstracted in a subclass of Monad.

import Control.Monad
import Control.Monad.Fail
import Data.List 
import System.IO
import System.Environment

--------------------------------------------------------------------------------
-- Primitives

class (Monad m, Desc d) => Circuit m d | m -> d where
  not1 :: Bit -> m Bit
  assert1 :: (Bit,Bool) -> m ()
  and2, or2, xor2 :: (Bit,Bit) -> m Bit
  mux3 :: (Bit,Bit,Bit) -> m Bit
  addDesc :: [d] -> m ()

class Desc d where 
  and2_desc :: Bit -> Bit -> Bit -> [d]
  or2_desc :: Bit -> Bit -> Bit -> [d]
  xor2_desc :: Bit -> Bit -> Bit -> [d]
  assert1_desc :: Bit -> Bool -> [d]
  mux3_desc :: Bit -> Bit -> Bit -> Bit -> [d]
  halfAdder_desc :: (Bit,Bit) -> (Bit,Bit) -> [d]
  fullAdder_desc :: (Bit,Bit,Bit) -> (Bit,Bit) -> [d]

type Var = Int

data Bit = Bool Bool | BitVar Var | NotBitVar Var deriving Eq

instance Show Bit where
  show (Bool True) = "True"
  show (Bool False) = "False"
  show (BitVar v) = show v
  show (NotBitVar v) = "-" ++ show v

low,high :: Bit
low = Bool False
high = Bool True

notBit :: Bit -> Bit
notBit (Bool b) = Bool (not b)
notBit (BitVar v) = NotBitVar v
notBit (NotBitVar v) = BitVar v

-- Least significant bit first, 6 ==> [low,high,high]
int2bits :: Integer -> [Bit]
int2bits 0 = []
int2bits n = let (d,m) = n `divMod` 2 in int2bit m : int2bits d

bits2int :: [Bit] -> Integer
bits2int [] = 0
bits2int (b:bs) = bit2int b + (2 * bits2int bs)

int2bit 1 = high
int2bit 0 = low

bit2int (Bool True) = 1
bit2int (Bool False) = 0

-- The built-in zip functions only zip up to the length of the
-- shortest list and ignore the remaining elements. Instead we 
-- add "false" to the end of the shorter lists. 

zipPad :: [Bit] -> [Bit] -> [(Bit,Bit)]
zipPad xs ys = 
  let size = max (length xs) (length ys)
      lows = repeat low
  in take size (zip (xs ++ lows) (ys ++ lows))

zip3Pad :: [Bit] -> [Bit] -> [Bit] -> [(Bit,Bit,Bit)]
zip3Pad xs ys zs = 
    let size = maximum (map length [xs,ys,zs])
        lows = repeat low 
    in take size (zip3 (xs ++ lows) (ys ++ lows) (zs ++ lows))

-- List

listby2 :: [a] -> ([(a,a)],[a])
listby2 [] = ([],[])
listby2 [a] = ([],[a])
listby2 (a1:a2:as) = 
    let (g,r) = listby2 as
    in ((a1,a2):g, r)

listby3 :: [a] -> ([(a,a,a)],[a])
listby3 [] = ([],[])
listby3 [a] = ([],[a])
listby3 [a1,a2] = ([],[a1,a2])
listby3 (a1:a2:a3:as) = 
    let (g,r) = listby3 as
    in ((a1,a2,a3):g, r)

-- Combinators

(>->) :: Circuit m d => (a -> m b) -> (b -> m c) -> (a -> m c)
(f >-> g) a = 
    do b <- f a
       g b

compose :: Circuit m d => [a -> m a] -> (a -> m a)
compose = foldr (>->) return

--------------------------------------------------------------------------------
-- Standard interpretation

data Std a = Std a

simulate :: Std a -> a
simulate (Std a) = a

instance Functor Std where
  fmap f (Std a) = Std (f a)

instance Applicative Std where
  pure = Std
  (Std f) <*> (Std a) = Std (f a)

instance Monad Std where
  (Std e1) >>= e2 = e2 e1

instance Desc () where
  and2_desc _ _ _ = [()]
  or2_desc _ _ _ = [()]
  xor2_desc _ _ _ = [()]
  assert1_desc _ _ = [()]
  mux3_desc _ _ _ _ = [()]
  halfAdder_desc _ _ = [()]
  fullAdder_desc _ _ = [()]

instance Circuit Std () where
  not1 (Bool x) = return (Bool (not x))
  and2 (Bool x, Bool y) = return (Bool (x && y))
  assert1 (Bool x, y) = if x == y then return () 
                        else error "Boolean assertion failed"
  or2 (Bool x, Bool y) = return (Bool (x || y))
  xor2 (Bool x, Bool y) = return (Bool (x /= y))
  mux3 (Bool False, a, b) = return a
  mux3 (Bool True, a, b) = return b
  addDesc _ = return ()

--------------------------------------------------------------------------------
-- Symbolic interpretations parameterized by a description of the gates

-- The description of a gate takes the output var, the input vars and 
-- returns a description of the gate

firstVar :: Var
firstVar = 1

class Circuit m d => Symbolic m d | m -> d where
  newBitVar :: m Bit        -- returns BitVar 1, BitVar 2, BitVar 3, ...

-- In symbolic execution, we carry a set of fresh names and a set of
-- generated descriptions. So a circuit computing a value of type a
-- is now of type (Sym desc a) below

data Sym desc a = Sym ([Var] -> (a, [Var], [desc]))

instance Functor (Sym desc) where
  fmap f (Sym g) = 
    Sym (\vars -> let (a, vars', desc) = g vars in (f a, vars', desc))

instance Applicative (Sym desc) where
  pure e = Sym (\ vars -> (e,vars,[]))
  (Sym c) <*> (Sym g) = 
    Sym (\vars -> let (f , vars', desc) = c vars
                      (a , vars'', desc') = g vars'
                  in (f a, vars'', desc++desc'))

instance Desc desc => Monad (Sym desc) where
  (Sym e1) >>= e2 =
      Sym (\ vars0 -> 
        let (v1,vars1,as1) = e1 vars0
            Sym f2 = e2 v1
            (v2,vars2,as2) = f2 vars1
        in (v2,vars2,as1++as2))

instance Desc desc => MonadFail (Sym desc) where
  fail _ = error "MonadFail: Internal bug"

-- In symbolic execution, the output of a gate is the name of its output variable.
-- or a constant boolean if we are able to simplify the circuit.
-- As a side-effect of execution a new description is added to the list we accumulate.

instance Desc desc => Circuit (Sym desc) desc where

  addDesc as = Sym (\ vars -> ((), vars, as))

  not1 (Bool x) = return (Bool (not x))
  not1 b = return (notBit b)

  and2 (Bool False, _) = return (Bool False)
  and2 (_, Bool False) = return (Bool False)
  and2 (Bool True, b) = return b
  and2 (b, Bool True) = return b
  and2 (b1, b2) = 
    do BitVar v <- newBitVar
       addDesc (and2_desc (BitVar v) b1 b2)
       return (BitVar v)

  or2 (Bool True, _) = return (Bool True)
  or2 (_, Bool True) = return (Bool True)
  or2 (Bool False, b) = return b
  or2 (b, Bool False) = return b
  or2 (b1, b2) = 
    do BitVar v <- newBitVar
       addDesc (or2_desc (BitVar v) b1 b2)
       return (BitVar v)                        

  xor2 (Bool True, b) = return (notBit b)
  xor2 (b, Bool True) = return (notBit b)
  xor2 (Bool False, b) = return b
  xor2 (b, Bool False) = return b
  xor2 (b1, b2) = 
    do BitVar v <- newBitVar
       addDesc (xor2_desc (BitVar v) b1 b2)
       return (BitVar v)                        

  assert1 (Bool b1, b2) = if b1 == b2 then return () 
                          else error "Boolean assertion failed"
  assert1 (v, b) = addDesc (assert1_desc v b)

  mux3 (Bool False, a, b) = return a
  mux3 (Bool True, a, b) = return b
  mux3 (s, Bool False, Bool False) = return (Bool False)
  mux3 (s, Bool False, Bool True) = return s
  mux3 (s, Bool True, Bool False) = not1 s
  mux3 (s, Bool True, Bool True) = return (Bool True)
  mux3 (s, Bool False, b) = and2(s,b)
  mux3 (s, Bool True, b) = do ns <- not1 s; or2(b,ns)
  mux3 (s, a, Bool False) = do ns <- not1 s; and2(ns,a)
  mux3 (s, a, Bool True) = or2(a,s)
  mux3 (s, a, b) = 
    do BitVar v <- newBitVar
       addDesc (mux3_desc (BitVar v) s a b)
       return (BitVar v)

instance Desc desc => Symbolic (Sym desc) desc where
  newBitVar = Sym (\ (v:vars) -> (BitVar v,vars,[]))

-- To run a circuit, we return the output, the last variable, 
-- and the list of generated descriptions

symbolic :: Desc desc => Sym desc a -> (a,Var,[desc])
symbolic (Sym f) = 
    let (a,nextVar:_,ds) = f [firstVar..]
    in (a,nextVar-1,ds)

--------------------------------------------------------------------------------
-- Description based on CNF formula

data CNFOr = CNFOr [Bit] | Com String

cnf_pred :: CNFOr -> Bool
cnf_pred (CNFOr _) = True
cnf_pred (Com _) = False

instance Desc CNFOr where

  assert1_desc v True = 
      [CNFOr [v],
      Com ("Assertion that " ++ show v ++ " is True")]

  assert1_desc v False = 
      [CNFOr [notBit v],
       Com ("Assertion that " ++ show v ++ " is False")]

  and2_desc v a b = 
      let or1 = CNFOr [a, b, notBit v]
          or2 = CNFOr [a, notBit b, notBit v]
          or3 = CNFOr [notBit a, b, notBit v]
          or4 = CNFOr [notBit a, notBit b, v]
       in [or1,or2,or3,or4,
         Com ("c And gate:\n" ++ 
              "c  inputs = " ++ show a ++ " and " ++ show b ++ "\n" ++ 
              "c  output = " ++ show v)]

  or2_desc v a b = 
      let or1 = CNFOr [a, b, notBit v]
          or2 = CNFOr [a, notBit b, v]
          or3 = CNFOr [notBit a, b, v]
          or4 = CNFOr [notBit a, notBit b, v]
      in [or1,or2,or3,or4,
         Com ("c Or gate:\n" ++ 
              "c  inputs = " ++ show a ++ " and " ++ show b ++ "\n" ++ 
              "c  output = " ++ show v)]

  xor2_desc v a b = 
      let or1 = CNFOr [a, b, notBit v]
          or2 = CNFOr [a, notBit b, v]
          or3 = CNFOr [notBit a, b, v]
          or4 = CNFOr [notBit a, notBit b, notBit v]
      in [or1,or2,or3,or4,
         Com ("c Xor gate:\n" ++ 
              "c  inputs = " ++ show a ++ " and " ++ show b ++ "\n" ++ 
              "c  output = " ++ show v)]

  mux3_desc v s a b = 
      let or1 = CNFOr [s, a, notBit v]
          or2 = CNFOr [s, notBit a, v]
          or3 = CNFOr [notBit s, b, notBit v]
          or4 = CNFOr [notBit s, notBit b, v]
      in [or1,or2,or3,or4,
          Com ("c Mux gate:\n" ++
               "c  inputs = " ++ show s ++ " and " ++ show a ++ " and " 
                   ++ show b ++ "\n" ++
               "c  output = " ++ show v)]

  halfAdder_desc (a,b) (carryOut,sum) = 
      [Com ("c Half adder:\n" ++ 
            "c   inputs = " ++ show a ++ " and " ++ show b ++ "\n" ++
            "c   carryOut = " ++ show carryOut ++ "\n" ++
            "c   sum = " ++ show sum)]

  fullAdder_desc (carryIn,a,b) (carryOut,sum) = 
      [Com ("c Full adder:\n" ++ 
            "c   carryIn = " ++ show carryIn ++ "\n" ++
            "c   inputs = " ++ show a ++ " and " ++ show b ++ "\n" ++
            "c   carryOut = " ++ show carryOut ++ "\n" ++
            "c   sum = " ++ show sum)]

instance Show CNFOr where
  show (CNFOr bits) = (concat (intersperse " " (map show bits))) ++ " 0"
  show (Com s) = s

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Now can write actual circuits, simulate them and generate formulae

--------------------------------------------------------------------
-- Shift (multiply by powers of two)
--------------------------------------------------------------------

-- shift 2 [0,0,1] ==> [0,0,0,0,1]
-- Because lsb comes first this is: shift 2 4 = 16
shift :: Int -> [Bit] -> [Bit]
shift n bs = replicate n low ++ bs

--------------------------------------------------------------------
-- Assert-n
--------------------------------------------------------------------

assert_n :: Circuit m d => [Bit] -> [Bool] -> m ()
assert_n bs vs = 
    zipWithM_ (curry assert1) bs (vs ++ (repeat False))

--------------------------------------------------------------------
-- Half adder
--------------------------------------------------------------------

-- halfAdd (b1,b2) = (carry,sum)
halfAdd :: Circuit m d => (Bit,Bit) -> m (Bit,Bit)
halfAdd (a,b) = 
  do carry <- and2 (a,b)
     sum <- xor2 (a,b)
     addDesc (halfAdder_desc (a,b) (carry,sum))
     return (carry,sum)

--------------------------------------------------------------------
-- 1-Bit full adder
--------------------------------------------------------------------

-- fullAdd (carryIn,b1,b2) = (carryOut,sum)
fullAdd :: Circuit m d => (Bit,Bit,Bit) -> m (Bit,Bit)
fullAdd (carryIn,a,b) = 
  do (carry1,sum1) <- halfAdd (a,b)
     (carry2,sum) <- halfAdd (carryIn, sum1)
     carryOut <- xor2 (carry1,carry2)
     addDesc (fullAdder_desc (carryIn,a,b) (carryOut,sum))
     return (carryOut,sum)

-- another version that is easier to compose
fullAddBlock :: Circuit m d => (Bit,[Bit]) -> (Bit,Bit) -> m (Bit,[Bit])
fullAddBlock (carryIn,prevSums) (a,b) = 
  do (carryOut,sum) <- fullAdd (carryIn,a,b)
     return (carryOut, prevSums ++ [sum])

--------------------------------------------------------------------
-- N-Bit adder
--------------------------------------------------------------------

type Adder m = ([Bit],[Bit]) -> m [Bit]
type AdderCarry m = ([Bit],[Bit],Bit) -> m ([Bit],Bit)

n_adder_carry :: Circuit m d => AdderCarry m
n_adder_carry (as,bs,carryIn) = 
  do (carryOut,sums) <- foldM fullAddBlock (carryIn,[]) (zipPad as bs)
     return (sums,carryOut)

n_adder :: Circuit m d => Adder m
n_adder (as,bs) = 
    do (sum,carryOut) <- n_adder_carry (as,bs,low)
       return (sum ++ [carryOut])

test_n_adder :: Integer -> Integer -> Integer
test_n_adder a b = 
    bits2int (simulate (n_adder (int2bits a, int2bits b)))

--------------------------------------------------------------------
-- Wallace-tree adder
--------------------------------------------------------------------

-- w_adder [bs1,bs2,...] = sum
w_adder :: Circuit m d => Adder m -> [[Bit]] -> m [Bit]
w_adder adder [] = return []
w_adder adder [as] = return as
w_adder adder [as,bs] = adder (as,bs)
w_adder adder xss = 
    let (groups,rest) = listby3 xss
    in do new_groups <- mapM w3_unit groups
          let (cs,ss) = unzip new_groups
          w_adder adder (map (low :) cs ++ ss ++ rest)
    where 
    w3_unit (as,bs,cs) = 
      do list_carries_sums <- mapM fullAdd (zip3Pad as bs cs)
         return (unzip list_carries_sums)

test_w_adder :: [Integer] -> Integer
test_w_adder xs = bits2int (simulate (w_adder n_adder (map int2bits xs)))

--------------------------------------------------------------------
-- n-bit and, or
--------------------------------------------------------------------

and_n :: Circuit m d => [Bit] -> m Bit
and_n = foldM (curry and2) high 

or_n :: Circuit m d => [Bit] -> m Bit
or_n = foldM (curry or2) low

--------------------------------------------------------------------
-- Multiplexer
--------------------------------------------------------------------

-- Now added as a primitive 
multiplex1 :: Circuit m d => (Bit,Bit,Bit) -> m Bit
multiplex1 (c,x,y) = 
    do notc <- not1 c
       select_first <- and2 (notc,x)
       select_second <- and2 (c,y)
       or2 (select_first,select_second)

-- n-bit multiplexer; both xs and ys should be of the same length
multiplex_n :: Circuit m d => (Bit,[Bit],[Bit]) -> m [Bit]
multiplex_n (c,xs,ys) = 
    mapM mux3 (zip3 (replicate (length xs) c) xs ys)

sym_multiplex :: IO ()
sym_multiplex = 
  let (_,_,f::[CNFOr]) = 
         symbolic (do c <- newBitVar
                      a <- newBitVar
                      b <- newBitVar
                      mux3 (c,a,b))
  in do putStr (concat (intersperse "\n" (map show f)))
        putStr "\n"

--------------------------------------------------------------------
-- Fast carry adder
--------------------------------------------------------------------

fast_adder_unit :: Circuit m d => (Bit,Bit) -> m ([Bit],[Bit],Bit,Bit)
fast_adder_unit (a,b) = 
    do s <- xor2 (a,b)    -- sum 
       t <- not1 s        -- sum with carry
       g <- and2 (a,b)    -- generate carry
       p <- or2 (a,b)     -- propagate carry
       return ([s],[t],g,p)

fast_adder :: Circuit m d => Adder m
fast_adder (as,bs) = 
  do us <- mapM fast_adder_unit (zipPad as bs)
     (s,_,g,_) <- apply_stages us
     return (s ++ [g])

  where apply_stages [u] = return u
        apply_stages us = (apply_stage >-> apply_stages) us

        apply_stage us = 
            let (groups,rest) = listby2 us
            in do wss <- mapM stage groups
                  return (wss ++ rest)

        stage ((s_low,t_low,g_low,p_low),(s_high,t_high,g_high,p_high)) = 
            do new_s_high <- multiplex_n (g_low,s_high,t_high)
               new_t_high <- multiplex_n (p_low,s_high,t_high)
               p <- do a <- or2 (g_high,p_high)
                       b <- or2 (g_high,p_low)
                       and2 (a,b)
               g <- do a <- or2 (g_high,p_high)
                       b <- or2 (g_high,g_low)
                       and2 (a,b)
               let s = s_low ++ new_s_high
                   t = t_low ++ new_t_high
               return (s,t,g,p)

test_f_adder :: Integer -> Integer -> Integer
test_f_adder a b = 
    bits2int (simulate (fast_adder (int2bits a, int2bits b)))

sym_f_adder :: IO ()
sym_f_adder = 
  let (_,_,f::[CNFOr]) = 
         symbolic (do a1 <- newBitVar
                      a2 <- newBitVar
                      b1 <- newBitVar
                      b2 <- newBitVar
                      fast_adder ([a1,a2],[b1,b2]))
  in do putStr (concat (intersperse "\n" (map show f)))
        putStr "\n"

--------------------------------------------------------------------
-- Subtraction
--------------------------------------------------------------------

-- works when xs >= ys
subt :: Circuit m d => ([Bit],[Bit]) -> m [Bit]
subt (xs,ys) = 
  let size = max (length xs) (length ys) 
      lows = repeat low
      exs = take size (xs ++ lows)
      eys = take size (ys ++ lows)
  in do zs <- mapM not1 eys -- one's complement
        carryIn <- not1 low
        (sum,carryOut) <- n_adder_carry (exs,zs,carryIn)
        assert1(carryOut,True)
        return sum

test_sub :: Integer -> Integer -> Integer 
test_sub a b = 
    bits2int (simulate (subt (int2bits a, int2bits b)))

--------------------------------------------------------------------
-- General multipliers
--------------------------------------------------------------------

type Multiplier m = ([Bit],[Bit]) -> m [Bit]

test_multiplier :: Multiplier Std -> Integer -> Integer -> Integer
test_multiplier m a b = 
    bits2int (simulate (m (int2bits a, int2bits b)))

type Sym_Multiplier = [Bit] -> ((([Bit],[Bit]),[Bit]),Var,[CNFOr])

sym_multiplier :: Multiplier (Sym CNFOr) -> Sym_Multiplier
sym_multiplier m output_bits = 
    let output_vals = map (\ (Bool b) -> b) output_bits
        output_size = genericLength output_bits
        input1_size = output_size - 1
        input2_size = (output_size + 1) `div` 2
    in symbolic 
        (do as <- mapM (const newBitVar) [1..input1_size]
            bs <- mapM (const newBitVar) [1..input2_size]
            cs' <- m (as,bs)
            let cs = filter keepBitVar cs'
            assert_n cs output_vals
            return ((as,bs),cs))
    where 
    -- some of the most significant bits might be the literal False
    -- get rid of those
    keepBitVar (Bool _) = False
    keepBitVar _ = True

--------------------------------------------------------------------
-- Recursive multiplier
--------------------------------------------------------------------

recursive_multiplier :: Circuit m d => Int -> Adder m -> Multiplier m
recursive_multiplier bound adder (xs,ys) = 
  let n = max (length xs) (length ys) 
      half_n = (n+1) `div` 2
  in if bound >= n
     then wallace_multiplier adder (xs,ys)
     else if length ys <= half_n -- cs below will be zero (specialize)
     then let lows = repeat low
              exs = take n (xs ++ lows)
              eys = take n (ys ++ lows)
              (bs,as) = splitAt half_n exs
              (ds,_) = splitAt half_n eys
          in do p3 <- recursive_multiplier bound adder (bs,ds)
                p2 <- recursive_multiplier bound adder (as,ds)
                adder (shift half_n p2, p3)
     else let lows = repeat low
              exs = take n (xs ++ lows)
              eys = take n (ys ++ lows)
              (bs,as) = splitAt half_n exs
              (ds,cs) = splitAt half_n eys
          in do p1 <- recursive_multiplier bound adder (as,cs)
                p3 <- recursive_multiplier bound adder (bs,ds)
                asbs <- adder (as,bs)
                csds <- adder (cs,ds)
                p2p1p3 <- recursive_multiplier bound adder (asbs,csds)
                p2p3 <- subt (p2p1p3,p1)
                p2 <- subt (p2p3,p3)
                s1 <- adder (shift (half_n*2) p1, shift half_n p2)
                adder (s1,p3)

--------------------------------------------------------------------
-- Carry-save Multiplier 
--------------------------------------------------------------------

carry_save_multiplier :: Circuit m d => Adder m -> Multiplier m
carry_save_multiplier adder (as,bs) = 
    let all_rows = map (\ b -> map (\ a -> block (a,b)) as) bs
        first_row = head all_rows
        rest_rows = tail all_rows
    in do first_results <- mapM (\ fa -> fa (low,low)) first_row
          (cs,last_results) <- middle_loop [] first_results rest_rows
          let (last_carries,last_sums) = unzip last_results
              shifted_sums = tail last_sums ++ [low]
          rest_cs <- adder(last_carries,shifted_sums)
          return (cs ++ [head last_sums] ++ rest_cs)
  where 
  block (x,y) (carryIn,sumIn) = 
      do xy <- and2 (x,y)
         fullAdd (carryIn,sumIn,xy)

  middle_loop cs previous_results [] = return (cs,previous_results)
  middle_loop cs previous_results rows_todo = 
    let (previous_carries,previous_sums) = unzip previous_results
        new_cs = cs ++ [head previous_sums]
        shifted_sums = tail previous_sums ++ [low]
        intermediate_inputs = zip previous_carries shifted_sums
    in do current_results <- zipWithM 
                               (\ fa (c,s) -> fa (c,s)) 
                               (head rows_todo)
                               intermediate_inputs
          middle_loop new_cs current_results (tail rows_todo)

--------------------------------------------------------------------
-- Wallace multiplier 
--------------------------------------------------------------------

wallace_multiplier :: Circuit m d => Adder m -> Multiplier m
wallace_multiplier adder (as,bs) = 
    do all_rows_aligned <- mapM (\ b -> mapM (\ a -> and2 (a,b)) as) bs
       let offsets = map (\n -> replicate n low) [0..]
           shifted_rows = zip offsets all_rows_aligned 
       w_adder adder (map (uncurry (++)) shifted_rows)

--------------------------------------------------------------------
-- Main
--------------------------------------------------------------------

-- bound for recursive multiplier
bound :: Int
bound = 20 -- should be more than 3 to avoid infinite loops ?

main :: IO ()
main = 
 do args <- getArgs
    if length args /= 3
       then error "Usage: cnf number [n-bit|fast] [carry-save|wallace|recursive]\n" 
       else return ()
    let output_num_dec = 
            let n = read (args !! 0) in 
            if n > 1 then n else error "Input number must be greater than 1"
        output_bits = int2bits output_num_dec
        adder = case args !! 1 of
                  "n-bit" -> n_adder
                  "fast" -> fast_adder
                  s -> error ("Adder " ++ s ++ " unknown")
        multiplier = sym_multiplier 
                       (case args !! 2 of 
                        "carry-save" -> carry_save_multiplier adder
                        "wallace" -> wallace_multiplier adder
                        "recursive" -> recursive_multiplier bound adder
                        s -> error ("Multiplier " ++ s ++ " unknown"))

        (((as,bs),cs),last,cnf) = multiplier output_bits

    putStr ("c Problem generated from " ++ (args !! 2) ++ "["
            ++ (args !! 1) ++ "] multiplication circuit\n")
    putStr "c by Paul Purdom and Amr Sabry\n"
    putStr "c\n"
    putStr ("c Circuit for product = " ++ show output_num_dec ++ 
            " " ++ (show (reverse output_bits)) ++ "\n")
    putStr "c Variables for output [msb,...,lsb]: "
    print (reverse cs)
    putStr "c Variables for first input [msb,...,lsb]: "
    print (reverse as)
    putStr "c Variables for second input [msb,...,lsb]: "
    print (reverse bs)
    putStr "c\nc\np cnf "
    putStr (show last ++ " " ++ show (length (filter cnf_pred cnf)) ++ "\n")
    putStr (concat (intersperse "\n" (map show (filter cnf_pred cnf))))
    putStr "\n"

--------------------------------------------------------------------