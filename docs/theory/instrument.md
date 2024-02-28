The instruments used to acquire X-ray spectra are modelled using the following quantities.

## Effective area

## Redistribution matrix

## Grouping matrix

## Expected counts

We can now write the expected counts in each bin as a function of the model :

$$
\begin{array}{c}
\boxed{
    \begin{array}{cc}
    \vdots \\
    \text{Expected} \\
    \text{counts} \\
    \vdots \\
    \end{array}
} \\
_{N_\text{Bins}}
\end{array}
= \begin{array}{c}
\boxed{
    \begin{array}{cc}
    & \vdots  & \\
    & \text{Binning} & \\
   \dots & \text{scheme} & \dots \\
    & \vdots  & \\
    \end{array}
} \\
_{N_\text{Bins} \times N_\text{Channels}}
\end{array} \times \begin{array}{c}
\boxed{
   \begin{array}{cc}
    & \vdots  & \\
   & \text{Redistribution} & \\
   \dots & \text{probability} & \dots \\
    & \vdots  & \\
    \end{array}
} \\
_{N_\text{Channels} \times N_\text{Entries}}
\end{array} \times \left( \begin{array}{c}
\boxed{
    \begin{array}{cc}
    \vdots \\
    \text{Integrated} \\
    \text{model} \\
    \vdots \\
    \end{array}
} \\
_{N_\text{Entries}}
\end{array} ~ \circ ~ \begin{array}{c}
\boxed{
    \begin{array}{cc}
    \vdots \\
    \text{Effective} \\
    \text{area} \\
    \vdots \\
    \end{array}
} \\
_{N_\text{Entries}}
\end{array} \times \text{Exposure}
\right)
$$

where $\times$ stands for the matrix product and $\circ$ for the element-wise (or Hadamard) product.
For a given instrumental set-up, the binning scheme, redistribution matrix, and effective area are fixed,
hence we can define a fixed transfer matrix that can be applied to any model :

$$
\begin{array}{c}
\boxed{
    \begin{array}{cc}
    & \vdots  & \\
    & \text{Transfer} & \\
   \dots & \text{Matrix} & \dots \\
    & \vdots  & \\
    \end{array}
} \\
_{N_\text{Bins} \times N_\text{Entries}}
\end{array}
= \begin{array}{c}
\boxed{
    \begin{array}{cc}
    & \vdots  & \\
    & \text{Binning} & \\
   \dots & \text{scheme} & \dots \\
    & \vdots  & \\
    \end{array}
} \\
_{N_\text{Bins} \times N_\text{Channels}}
\end{array} \times \begin{array}{c}
\boxed{
   \begin{array}{cc}
    & \vdots  & \\
   & \text{Redistribution} & \\
   \dots & \text{probability} & \dots \\
    & \vdots  & \\
    \end{array}
} \\
_{N_\text{Channels} \times N_\text{Entries}}
\end{array} \times \begin{array}{c}
\boxed{
    \begin{array}{cc}
    \vdots \\
    \text{Effective} \\
    \text{area} \\
    \vdots \\
    \end{array}
} \\
_{N_\text{Entries}}
\end{array} \times \text{Exposure}
$$