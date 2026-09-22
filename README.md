**HiBASIL: Hierarchical BAyesian Source Identification & Localization**
 
HiBASIL is a hierarchical Bayesian framework designed to solve the "inverse problem" of biological dispersal: locating unknown sources and estimating dispersal kernels from overlapping, mixed signals.

🚀 **Why HiBASIL?**

Decoupling mixed spatial signals constitutes a severely "ill-posed" inverse problem. Separating mixed vector contributions and parameter estimation in the Total Dispersal Kernel (TDK) are statistically challenged (Rogers et al., 2019). HiBASIL explicitly models the geometric superposition of multiple sources, allowing users to disentangle "spatial mixtures" even when data is bounded or zero- and one-inflated.

✨ **Key Capabilities**

**Unknown Source Localization**: Pinpoints coordinates $(x, y)$ of unknown sources.

**Multi-Source Disentangle**: Resolves the relative contribution $(w)$ of overlapping infection foci.

**Mechanistic Fidelity**: Fits biologically realistic dispersal kernels (Exponential, Power Law, Gaussian) while handling Zero- and One-Inflation (ZOIB).

📊 **Performance at a Glance**

Below: HiBASIL resolving a two-foci scenarios from unknonw locations. The framework successfully deconvolves the spatial mixture to identify the two primary sources.

<figure>
  <img src="image/6040twofoci.png" alt="Description">
  <figcaption>Figure 1: Spatial validation of posterior predictive distribution at foci mixture weights (0.6,0.4). Main panels present 1D spatial transects at x = 0, comparing observed severity (black points) against the posterior predictive mean (navy line) and 95% credible interval (shaded sky-blue region). The vertical dashed lines denote the estimated source locations at y = 0 and y = 50. The 2D insets demonstrate the framework’s capacity for joint estimation, successfully localizing unknown infection sources (stars) relative to
the generative true foci (crosses) using weakly informative priors.</figcaption>
</figure>


📖 **Tutorials & Documentation**

We provide standalone, documented scripts to guide users through different levels of spatial complexity:

1. [Single-Focus Localization]: The "Needle in a Haystack" tutorial. Learns to find one unknown origin and its decay parameters.

2. [Two-Foci Resolution]: The "Spatial Mixture" tutorial. Learns to separate two verlapping signals and infer their mixture weights.

3. [Epicenter of the Historical Cholera Outbreak]: The "Real-World Mystery" scenario. Learn to use HiBASIL to detect the epicenter in complex Human society environment.

