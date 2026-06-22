# 🔀 GIT - GÉRER PLUSIEURS REMOTES

**Problème** : J'ai mon repo perso et je veux aussi push sur le repo du groupe  
**Solution** : Ajouter plusieurs "remotes" Git

---

## 📚 CONCEPTS

**Remote** = Lien vers un repo Git distant (GitHub, GitLab, etc.)

Par défaut, tu as :
- `origin` = Ton repo perso

On va ajouter :
- `group` = Repo du groupe

---

## ⚙️ CONFIGURATION

### 1. Voir tes remotes actuels

```bash
git remote -v
```

**Résultat actuel** :
```
origin  https://github.com/TON_USER/zoidberg.git (fetch)
origin  https://github.com/TON_USER/zoidberg.git (push)
```

---

### 2. Ajouter le remote du groupe

**Si le groupe a déjà créé le repo** :

```bash
# Ajouter le remote groupe
git remote add group https://github.com/GROUPE_ORG/zoidberg-projet.git

# Vérifier
git remote -v
```

**Résultat** :
```
origin  https://github.com/TON_USER/zoidberg.git (fetch)
origin  https://github.com/TON_USER/zoidberg.git (push)
group   https://github.com/GROUPE_ORG/zoidberg-projet.git (fetch)
group   https://github.com/GROUPE_ORG/zoidberg-projet.git (push)
```

---

### 3. Pusher sur les 2 repos

#### Option A : Push manuel sur chaque remote

```bash
# Push sur ton repo perso
git push origin feature/multiclass-v3

# Push sur le repo groupe
git push group feature/multiclass-v3
```

#### Option B : Push automatique sur les 2 (recommandé)

```bash
# Configurer origin pour pusher sur les 2
git remote set-url --add --push origin https://github.com/TON_USER/zoidberg.git
git remote set-url --add --push origin https://github.com/GROUPE_ORG/zoidberg-projet.git

# Maintenant un seul push suffit
git push origin feature/multiclass-v3
# → Pushera sur les DEUX repos !
```

#### Option C : Créer un alias pour pusher partout

```bash
# Créer un alias "pushall"
git config alias.pushall '!git push origin && git push group'

# Utilisation
git pushall
# → Pushe sur origin ET group
```

---

## 🎯 WORKFLOWS RECOMMANDÉS

### Workflow 1 : Perso d'abord, Groupe après (SAFE)

**Utilisation** : Développement personnel puis partage au groupe

```bash
# 1. Développer localement
git add .
git commit -m "feat: add something"

# 2. Push sur TON repo (test)
git push origin feature/multiclass-v3

# 3. Vérifier que tout marche

# 4. Push sur le GROUPE
git push group feature/multiclass-v3
```

**Avantages** :
- ✅ Tu testes d'abord sur ton repo
- ✅ Pas de pollution du repo groupe avec WIP
- ✅ Contrôle total

---

### Workflow 2 : Sync automatique (FAST)

**Utilisation** : Collaboration active, tout est partagé

```bash
# Configurer push automatique (fait une fois)
git remote set-url --add --push origin https://github.com/GROUPE_ORG/zoidberg-projet.git

# Ensuite, push normal
git push origin main
# → Pushe sur TON repo ET le groupe automatiquement
```

**Avantages** :
- ✅ Rapide (1 commande)
- ✅ Toujours en sync

**Inconvénients** :
- ⚠️ Tout est partagé immédiatement (même WIP)

---

### Workflow 3 : Branches séparées (FLEXIBLE)

**Utilisation** : Ton travail perso + branche groupe

```bash
# Ton travail perso sur ta branche
git checkout feature/my-work
git push origin feature/my-work  # Seulement sur ton repo

# Travail partagé sur branche commune
git checkout main
git merge feature/my-work  # Merger ton travail
git push group main        # Push groupe seulement
```

---

## 📋 COMMANDES UTILES

### Voir les remotes
```bash
git remote -v
```

### Ajouter un remote
```bash
git remote add <name> <url>
```

### Supprimer un remote
```bash
git remote remove <name>
```

### Renommer un remote
```bash
git remote rename <old-name> <new-name>
```

### Voir les infos d'un remote
```bash
git remote show origin
```

### Pull depuis un remote spécifique
```bash
git pull group main
```

### Fetch depuis tous les remotes
```bash
git fetch --all
```

---

## 🔐 AUTHENTIFICATION

### Si le repo groupe utilise SSH

```bash
# Ajouter avec URL SSH
git remote add group git@github.com:GROUPE_ORG/zoidberg-projet.git
```

**Avantage** : Pas besoin de mot de passe à chaque push

### Si le repo groupe utilise HTTPS

```bash
# Ajouter avec URL HTTPS
git remote add group https://github.com/GROUPE_ORG/zoidberg-projet.git
```

**Note** : Peut demander identifiants à chaque push (configurer credential helper)

```bash
# Activer le cache de credentials (15 min)
git config --global credential.helper cache

# Ou stocker credentials (attention sécurité)
git config --global credential.helper store
```

---

## 🚀 SCÉNARIOS PRATIQUES

### Scénario 1 : Premier push sur le repo groupe

```bash
# 1. Ton collègue a créé le repo groupe
# 2. Ajouter le remote
git remote add group https://github.com/GROUPE_ORG/zoidberg-projet.git

# 3. Push ta branche actuelle
git push group feature/multiclass-v3

# 4. Créer un pull request sur le repo groupe
# (via l'interface GitHub du repo groupe)
```

---

### Scénario 2 : Synchroniser avec le groupe

```bash
# Récupérer les changements du groupe
git fetch group

# Voir les branches du groupe
git branch -r | grep group

# Merger les changements du groupe dans ta branche
git pull group main

# Résoudre conflits si besoin

# Push sur ton repo perso
git push origin main
```

---

### Scénario 3 : Contribuer à une branche commune

```bash
# 1. Récupérer la branche du groupe
git fetch group
git checkout -b dev-group group/dev

# 2. Faire tes modifications
git add .
git commit -m "feat: my contribution"

# 3. Push sur le groupe
git push group dev-group
```

---

## ⚙️ CONFIGURATION AVANCÉE

### Push sur 2 repos avec 1 commande

**Méthode 1 : Modifier origin**
```bash
# Configurer origin pour pusher sur 2 URLs
git remote set-url --add --push origin https://github.com/TON_USER/zoidberg.git
git remote set-url --add --push origin https://github.com/GROUPE_ORG/zoidberg-projet.git

# Push
git push origin main  # → Pushe sur les 2 !
```

**Méthode 2 : Créer remote "all"**
```bash
# Créer un remote spécial qui push sur tous
git remote add all https://github.com/TON_USER/zoidberg.git
git remote set-url --add --push all https://github.com/TON_USER/zoidberg.git
git remote set-url --add --push all https://github.com/GROUPE_ORG/zoidberg-projet.git

# Push partout
git push all main
```

**Vérifier la config** :
```bash
cat .git/config

# Résultat :
[remote "all"]
    url = https://github.com/TON_USER/zoidberg.git
    pushurl = https://github.com/TON_USER/zoidberg.git
    pushurl = https://github.com/GROUPE_ORG/zoidberg-projet.git
```

---

## 📝 EXEMPLE COMPLET

```bash
# État actuel
cd c:\Users\nbrodbeck\Documents\code\zoidberg
git remote -v
# origin  https://github.com/nbrodbeck/zoidberg.git (fetch)
# origin  https://github.com/nbrodbeck/zoidberg.git (push)

# Ajouter le repo du groupe
git remote add group https://github.com/groupe-intech/zoidberg-pneumonia.git

# Vérifier
git remote -v
# origin  https://github.com/nbrodbeck/zoidberg.git (fetch)
# origin  https://github.com/nbrodbeck/zoidberg.git (push)
# group   https://github.com/groupe-intech/zoidberg-pneumonia.git (fetch)
# group   https://github.com/groupe-intech/zoidberg-pneumonia.git (push)

# Push sur ton repo perso
git push origin feature/multiclass-v3

# Push sur le repo groupe
git push group feature/multiclass-v3

# OU créer un alias pour les 2
git config alias.pushall '!git push origin && git push group'
git pushall
```

---

## ✅ CHECKLIST

Avant de pusher sur le repo groupe :

- [ ] Vérifier qu'il n'y a pas de données sensibles (.env, credentials)
- [ ] Vérifier .gitignore (data/, venv/, models/ bien exclus)
- [ ] Commit messages clairs
- [ ] Code fonctionnel (tests passent)
- [ ] Documentation à jour
- [ ] Pas de fichiers trop lourds (modèles .keras exclus)

---

## 🆘 TROUBLESHOOTING

### Erreur : "Permission denied"
```bash
# Le groupe ne t'a pas donné accès
# → Demander à être ajouté comme collaborateur
```

### Erreur : "Repository not found"
```bash
# URL incorrecte
# Vérifier l'URL exacte du repo groupe
git remote set-url group https://CORRECT_URL.git
```

### Conflits lors du pull
```bash
# Récupérer les changements du groupe
git fetch group

# Voir les différences
git diff main group/main

# Merger avec résolution manuelle
git merge group/main
# Résoudre conflits
git add .
git commit -m "merge: resolve conflicts with group repo"
```

---

## 🎯 RECOMMANDATION POUR TON PROJET

**Setup initial** (fait une fois) :

```bash
# 1. Ajouter le remote groupe (remplacer URL)
git remote add group https://github.com/GROUPE_ORG/zoidberg-projet.git

# 2. Créer un alias pour push facile
git config alias.pushboth '!git push origin $1 && git push group $1'

# 3. Usage
git pushboth feature/multiclass-v3
# → Pushe sur TON repo ET le groupe
```

---

**📚 Ressources** :
- Git Doc Remotes : https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes
- GitHub Multiple Remotes : https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories

*Document créé le 22 juin 2026*
