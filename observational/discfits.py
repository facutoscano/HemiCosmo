"""
Standard file format for disc-by-disc parameter fits (one .npz file).

  params        float [n_real, n_disc, n_par]   best-fit parameters per disc
                realisation 0 = the data map, 1..n_real-1 = isotropic simulations
                processed with the SAME pipeline (mask, multipole range, fit)
  param_names   str   [n_par]   must include H0, ombh2, omch2 (aliases below accepted)
  nside         int             disc centres = HEALPix RING pixel centres at this nside
  disc_radius_deg  float        angular radius of the discs (FG2021/T2025: 30 = 60 deg diameter)
  valid         bool  [n_real, n_disc]  optional: False for failed / discarded fits
  source        str             optional free text (pipeline, data release, date)

Write it from your own fit outputs with write_discfits(); everything downstream only
reads this format. Example (T2025 fits stored as one text file per realisation):

    params = np.stack([np.loadtxt(f) for f in files])        # [n_real, 192, n_par]
    write_discfits("data/discfits_T2025_D60.npz", params,
                   ["omch2", "ombh2", "H0", "ns", "As", "A_PS"], nside=4,
                   disc_radius_deg=30, source="T2025 pipeline, SMICA PR3")
"""
import numpy as np

ALIASES = {"omegach2": "omch2", "omega_c": "omch2", "wc": "omch2", "Omegach2": "omch2",
           "omegabh2": "ombh2", "omega_b": "ombh2", "wb": "ombh2", "Omegabh2": "ombh2",
           "h0": "H0"}
REQUIRED = ("H0", "ombh2", "omch2")


def canonical(name: str) -> str:
    return ALIASES.get(name, ALIASES.get(name.lower(), name))


class DiscFits:
    def __init__(self, params, names, nside, disc_radius_deg, valid=None, source=""):
        self.params = np.asarray(params, float)
        self.names = [canonical(str(n)) for n in names]
        self.nside = int(nside)
        self.disc_radius_deg = float(disc_radius_deg)
        n_real, n_disc, n_par = self.params.shape
        if n_disc != 12 * self.nside ** 2:
            raise ValueError(f"{n_disc} discs but nside={self.nside} has {12 * self.nside ** 2} pixels")
        if n_par != len(self.names):
            raise ValueError(f"{n_par} parameters but {len(self.names)} names")
        missing = [r for r in REQUIRED if r not in self.names]
        if missing:
            raise ValueError(f"disc fits lack {missing} (have {self.names}); "
                             f"ln theta_* needs H0, ombh2 and omch2 in EVERY realisation")
        v = np.ones((n_real, n_disc), bool) if valid is None else np.asarray(valid, bool)
        self.valid = v & np.all(np.isfinite(self.params), axis=2)
        self.source = str(source)

    @property
    def n_real(self) -> int:
        return self.params.shape[0]

    @property
    def n_sims(self) -> int:
        return self.n_real - 1

    def get(self, name: str) -> np.ndarray:
        """[n_real, n_disc] values of one parameter"""
        return self.params[:, :, self.names.index(canonical(name))]

    def summary(self) -> str:
        frac = self.valid.mean()
        return (f"{self.n_real} realisations (1 data + {self.n_sims} sims), "
                f"{self.params.shape[1]} discs (nside {self.nside}, radius "
                f"{self.disc_radius_deg:g} deg), params {self.names}, valid {frac:.1%}"
                + (f", source: {self.source}" if self.source else ""))


def write_discfits(path, params, names, nside, disc_radius_deg, valid=None, source=""):
    d = DiscFits(params, names, nside, disc_radius_deg, valid, source)      # validates
    np.savez_compressed(path, params=d.params, param_names=np.array(d.names),
                        nside=d.nside, disc_radius_deg=d.disc_radius_deg,
                        valid=d.valid, source=d.source)
    return path


def load_discfits(path) -> DiscFits:
    z = np.load(path, allow_pickle=False)
    return DiscFits(z["params"], [str(x) for x in z["param_names"]], int(z["nside"]),
                    float(z["disc_radius_deg"]),
                    z["valid"] if "valid" in z.files else None,
                    str(z["source"]) if "source" in z.files else "")
