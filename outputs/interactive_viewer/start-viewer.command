#!/bin/zsh
cd "${0:A:h}"
if [[ ! -d node_modules ]]; then
  npm install || exit 1
fi
npm run dev -- --open
