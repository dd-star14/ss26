import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("C:/java/Cas3_total.csv")
vr = df[df['P(member)']==1]['vr'].dropna()
n = len(vr)
print("Number of member stars =", n)

def log_likelihood(vr_model, sigma_model):

    log_pdf_each = (
        -0.5*np.log(2*np.pi)
        -np.log(sigma_model)
        -(vr-vr_model)**2/(2*sigma_model**2)
    )
    logL = log_pdf_each.sum()
    return logL

vr_current = -365
sigma_current = 13
logL_current = log_likelihood( vr_current,sigma_current)
num_steps = 50000
delta_vr = 2
delta_sigma = 1
chain_vr = np.zeros(num_steps)
chain_sigma = np.zeros(num_steps)
chain_logL = np.zeros(num_steps)
accept_count = 0
for i in range(num_steps):
    chain_vr[i] = vr_current
    chain_sigma[i] = sigma_current
    chain_logL[i] = logL_current
    vr_new = np.random.normal(loc=vr_current,scale=delta_vr)
    sigma_new = np.random.normal(loc=sigma_current, scale=delta_sigma)
    if sigma_new <= 0:
        continue
    logL_new = log_likelihood(vr_new,sigma_new)
    ratio = np.exp(logL_new-logL_current)
    if logL_new > logL_current:
        accept = True
    else:
        random_number = np.random.uniform(0,1)
        if random_number < ratio:
            accept = True
        else:
            accept = False
    if accept:
        vr_current = vr_new
        sigma_current = sigma_new
        logL_current = logL_new
        accept_count += 1
acceptance_rate = accept_count / num_steps

print(
    "Acceptance rate =",
    acceptance_rate
)


burn_in = int(num_steps*0.1)
vr_chain = chain_vr[burn_in:]
sigma_chain = chain_sigma[burn_in:]
logL_chain = chain_logL[burn_in:]
chain = pd.DataFrame(
    {
        "vr":vr_chain,
        "sigma":sigma_chain,
        "logL":logL_chain
    }
)

chain.to_csv(
    "Cas3_MCMC_chain.txt",
    index=False
)
print("Chain saved")


vr16, vr50, vr84 = np.percentile(vr_chain,[16,50,84])
sigma16, sigma50, sigma84 = np.percentile(sigma_chain,[16,50,84])
print(f"vr = {vr50:.2f} +{vr84-vr50:.2f} -{vr50-vr16:.2f}")
print(f"sigma = {sigma50:.2f} +{sigma84-sigma50:.2f} -{sigma50-sigma16:.2f}")

plt.figure(figsize=(10,4))
plt.plot(vr_chain)
plt.xlabel("Iteration")
plt.ylabel(r"$v_r$ (km/s)")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(sigma_chain)
plt.xlabel("Iteration")
plt.ylabel(r"$\sigma_v$ (km/s)")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(9,5))
plt.hist(
    vr_chain,
    bins=50,
    alpha=0.6
)
plt.axvline(
    vr50,
    linestyle="--",
    label=f"median={vr50:.2f}"
)
plt.xlabel(r"$v_r$ (km/s)")
plt.ylabel("sample number")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(9,5))
plt.hist(
    sigma_chain,
    bins=50,
    alpha=0.6
)
plt.axvline(
    sigma50,
    linestyle="--",
    label=f"median={sigma50:.2f}"
)
plt.xlabel(r"$\sigma_v$ (km/s)")
plt.ylabel("sample number")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


iteration = np.arange(len(chain_vr))
plt.figure(figsize=(10,8))
plt.scatter(
    chain_vr[:burn_in],
    chain_sigma[:burn_in],
    color="firebrick",
    s=18,
    alpha=0.5,
    label="Burn-in"
)
sc = plt.scatter(
    chain_vr[burn_in:],
    chain_sigma[burn_in:],
    c=iteration[burn_in:],
    cmap="jet",
    s=10,
    label="Posterior"
)
plt.scatter(
    chain_vr[-1],
    chain_sigma[-1],
    s=140,
    facecolors="none",
    edgecolors="black",
    linewidths=2
)
plt.xlabel("mean_vr")
plt.ylabel("sigma_vr")
cbar = plt.colorbar(sc)
cbar.set_label("Iteration_nb")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
