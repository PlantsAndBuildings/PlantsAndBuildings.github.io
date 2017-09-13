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

**Lemma 1**: If $$ G $$ is a group, then the following holds:

* The identity element in G is unique
* The inverse for every element in G is unique
* $$ (a^{-1})^{-1} = a\ \forall a\in G $$
* $$ (ab)^{-1} = b^{-1}a^{-1}\ \forall a,b \in G $$


<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

