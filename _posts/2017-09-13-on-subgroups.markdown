---
layout: post
title: On Subgroups
published: false
categories: [math, group theory, abstract algebra]
---

Okay, no time to waste! Lets begin with a review of groups before we get to subgroups and the good Langrangian stuff.

#### Preliminaries: Groups and other definitions

A set of elements $$G$$ is a group, if there is a binary operator ($$\circ$$), referred to henceforth as the **group operator**, defined on the elements of $$G$$ that satisfies the following properties:

* $$ a,b \in G \implies a \circ b \in G $$. (Closure).
* $$ a,b,c \in G \implies (a \circ b) \circ c = a \circ (b \circ c) $$. (Associativity).
* There exists an element $$ e \in G $$ such that $$ a \circ e = e \circ a = a,\  \forall a \in G $$. (Existence of identity).
* For each element $$ a \in G $$, there exists an element $$ a^{-1} \in G $$ such that $$ a \circ a^{-1} = a^{-1} \circ a = e $$. (Existence of inverse).

Closure, Associativity, Existence of identity and Existence of inverse. Boom! That's all it takes to get a group. Additionally, if the the group operator satisfies commutativity (Ie, $$ a \circ b = b \circ a\  \forall a,b \in G $$), then the group is said to be an **Abelian group** or a **Commutative group**.

The number of elements in a group is called the **order of the group** and is denoted by $$ o(G) $$.

**Symmetric Group of degree $$ n $$**: Let $$ S $$ denote a set having $$ n $$ elements. Let $$ A(S) $$ denote the set of all possible bijections from S onto itself. Also, we define the group operator to be the composition operator (Recall that a composition of functions $$ f:A \rightarrow B $$ and $$ g:B \rightarrow C $$ is given as $$ f \circ g: A \rightarrow C $$). It can be shown easily that this is a valid group and is in fact a special group - known as a symmetric group of degree $$ n $$. It is denoted as $$ S_{n} $$

We now look at two lemmas (and discuss their proofs in brief) - these may prove necessary for our later discussion on subgroups.

**Lemma 1**: If $$ G $$ is a group, then the following hold:

1. The identity element in G is unique
2. The inverse for every element in G is unique
3. \$$ (a^{-1})^{-1} = a,\ \forall a\in G $$
4. \$$ (ab)^{-1} = b^{-1}a^{-1},\ \forall a,b \in G $$

**Proof**:

1. If possible, let $$ e_1 $$ and $$ e_2 $$ be two distinct identity elements in $$ G $$. Thus,\\
\\
$$ a \circ e_1 = e_1 \circ a = a,\ \forall a \in G \tag{1} $$\\
$$ b \circ e_2 = e_2 \circ b = b,\ \forall b \in G \tag{2} $$\\
Since, $$ e_2 \in G $$, set $$ a $$ to be $$ e_2 $$ in $$ (1) $$\\
Since, $$ e_1 \in G $$, set $$ b $$ to be $$ e_1 $$ in $$ (2) $$\\
Thus, we have\\
\\
$$ e_2 \circ e_1 = e_1 \circ e_2 = e_2 \tag{3}$$\\
$$ e_1 \circ e_2 = e_2 \circ e_1 = e_1 \tag{4}$$\\
Which means that $$ e_1 = e_2 $$. Contradiction!
2. Consider the following\\
$$ a \circ x = a \circ y \tag{1} $$\\
We know that there exists some $$ b \in G $$ such that $$ a \circ b = b \circ a = e $$. For all we know, there might be several such $$ b $$s. Let us take any one (of possible many) $$ b $$ and apply the group operator to equation $$ (1) $$ as follows:\\
$$ \implies b \circ (a \circ x) = b \circ (a \circ y) \tag{2} $$\\
Applying associativity rule\\
$$ \implies (b \circ a) \circ x = (b \circ a) \circ y $$\\
$$ \implies x = y $$\\
Thus, we have $$ a \circ x = a \circ y \implies x = y $$. That is, common terms can be cancelled **if they appear on the same side**. We can now use this intermediate result to prove that for each element in $$ G $$, the inverse is unique. Suppose, that for some $$ a \in G $$, there exist $$ b_1, b_2 \in G $$ such that $$ b_1 \neq b_2 $$ and $$ a \circ b_1 = b_1 \circ a = e = a \circ b_2 = b_2 \circ a $$. Thus, by the cancellation property, $$ b_1 = b_2 $$.
3. We know that\\
$$ a^{-1} \circ (a^{-1})^{-1} = e \tag{1} $$\\
Also,\\
$$ a^{-1} \circ a = e \tag{2} $$\\
Equating left-hand sides of $$ (1) $$ and $$ (2) $$, we get\\
$$ a^{-1} \circ (a^{-1})^{-1} = a^{-1} \circ a $$\\
Using the cancellation property proven in the previous part, we get\\
$$ (a^{-1})^{-1} = a \tag{3} $$
4. This one's easy. Consider,\\
$$ (a \circ b) \circ (b^{-1} \circ a^{-1}) $$\\
$$ = a \circ (b \circ b^{-1}) \circ a^{-1} $$\\
$$ = a \circ e \circ a^{-1} $$\\
$$ = a \circ a^{-1} $$\\
$$ = e $$\\
Thus, $$ (a \circ b) \circ (b^{-1} \circ a^{-1}) = e $$. Hence, $$ (a \circ b)^{-1} = b^{-1} \circ a^{-1} $$

#### Subgroups

Suppose $$ G $$ is a group and $$ H $$ is a subset of $$ G $$ such that $$ H $$ itself is a group (under the same group operator as $$ G $$). Then, $$ H $$ is a **subgroup** of $$ G $$.

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

