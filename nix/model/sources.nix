# Resolve the declared documentation sources into store paths and manifests.
#
# `sources.json` is the hand-maintained declaration; `sources.lock.json` holds
# the revision and hash for each one (refreshed by `nix run .#update-sources`).
# Nothing here fetches anything unpinned, so a corpus build is reproducible and
# an out-of-date lock file is visible rather than silent.
{
  lib,
  fetchgit,
  fetchFromGitHub,
  writeText,
  nixpkgsSrc,
  declaration,
  lockfile,
}:

let
  decl = builtins.fromJSON (builtins.readFile declaration);
  lock = (builtins.fromJSON (builtins.readFile lockfile)).sources or { };

  cloneUrl =
    source:
    let
      host =
        {
          github = "https://github.com";
          codeberg = "https://codeberg.org";
          gitlab = "https://gitlab.com";
        }
        .${source.forge or "github"};
      # A GitHub wiki is a separate git repository. Niri's documentation lives
      # in one, which is the only way to get its current docs.
      suffix = if source.wiki or false then ".wiki.git" else ".git";
    in
    "${host}/${source.owner}/${source.repo}${suffix}";

  # Fetch one source, or return null when the lock file has no entry for it.
  # Returning null (rather than throwing) is what lets an optional source drop
  # out of the corpus without breaking the build.
  fetchSource =
    source:
    let
      entry = lock.${source.name} or null;
    in
    if source ? from then
      {
        root = "${nixpkgsSrc}/${source.subdir}";
        rev = nixpkgsSrc.rev or "unlocked";
      }
    else if entry == null then
      null
    else
      {
        inherit (entry) rev;
        root =
          if source ? sparse then
            fetchgit {
              name = "${source.name}-docs";
              url = cloneUrl source;
              inherit (entry) rev hash;
              sparseCheckout = source.sparse;
            }
          else if (source.forge or "github") == "github" && !(source.wiki or false) then
            fetchFromGitHub {
              inherit (source) owner repo;
              inherit (entry) rev hash;
            }
          else
            fetchgit {
              name = "${source.name}-docs";
              url = cloneUrl source;
              inherit (entry) rev hash;
            };
      };

  resolve =
    source:
    let
      fetched = fetchSource source;
    in
    if fetched == null then
      null
    else
      {
        inherit (source) name license;
        usage = source.usage or [ "retrieval" ];
        optional = source.optional or false;
        homepage = source.homepage or (lib.removeSuffix ".git" (cloneUrl source));
        description = source.description or "";
        include =
          source.include or [
            "**/*.md"
            "**/*.markdown"
            "**/*.rst"
            "**/*.txt"
          ];
        exclude = source.exclude or [ ];
        url_template = source.urlTemplate or "";
        inherit (fetched) rev;
        root = toString fetched.root;
      };

  resolved = lib.filter (s: s != null) (map resolve decl.sources);

  missing = lib.filter (s: !(s ? from) && !(lock ? ${s.name}) && !(s.optional or false)) decl.sources;

  # A required source with no lock entry is a bug in the lock file, not
  # something to paper over at build time.
  checked =
    if missing == [ ] then
      resolved
    else
      throw ''
        nix-llama: ${toString (builtins.length missing)} required documentation source(s) are
        missing from nix/model/sources.lock.json: ${lib.concatMapStringsSep ", " (s: s.name) missing}
        Run `nix run .#update-sources` to refresh it.
      '';

  manifestFields =
    s:
    builtins.removeAttrs s [
      "usage"
      "optional"
    ];

  forUsage = want: lib.filter (s: lib.elem want s.usage) checked;

  mkManifest =
    name: entries: writeText name (builtins.toJSON { sources = map manifestFields entries; });
in
{
  # Everything indexable, with its licence and revision.
  all = checked;

  # Sources permissively licensed enough to fine-tune on.
  training = forUsage "training";

  retrievalManifest = mkManifest "nix-llama-retrieval-sources.json" (forUsage "retrieval");
  trainingManifest = mkManifest "nix-llama-training-sources.json" (forUsage "training");

  # Human-readable summary, handy for `nix eval .#...` and for the model card.
  summary = map (s: {
    inherit (s)
      name
      license
      rev
      description
      ;
    inherit (s) usage;
  }) checked;
}
