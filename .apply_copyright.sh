#!/bin/bash 

# Name of the library/package/program to copyright
name="Stream"

# Year to update copyright
start_year="2025"
current_year=$(date +"%Y")

# Do not put the copyright notice on these files
ignore_files=(
    "AUTHORS.TXT"
    "COPYING.TXT"
    "LICENSE.TXT"
    "README.md"
    ".apply_copyright.sh"
    ".gitattributes"
    ".gitignore"
    ".gitmodules"
    "deps/"
    ".zip"
    ".dat"
    ".csv"
)

# The copyright notice to put on each file
copyright="
${name} - Copyright (C) <${start_year}-${current_year}>
<Universite catholique de Louvain (UCL), Belgique>

List of the contributors to the development of Stream: see AUTHORS file.
Description and complete License: see LICENSE file.

This file is part of ${name}. ${name} is free software:
you can redistribute it and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

${name} is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with ${name}. 
If not, see <https://www.gnu.org/licenses/>.
"

# Get the size of the copyright
copyright_nlines=$(wc -l <<< $copyright)

# Function to comment the copyright according to the file type
function comment_copyright {
    extension="$1"
    case $extension in 
        # Add a '# ' comment at the beginning of every line + escape newlines
        "py" | "txt" | "in") 
            echo "$(sed -e 's/^/# /g;$!s/$/\\/' <<< $copyright)";;
        # Add a '// ' comment at the beginning of every line + escape newlines
        "h" | "hpp" | "c" | "cpp" | "geo") 
            echo "$(sed -e 's/^/\/\/ /g;$!s/$/\\/' <<< $copyright)";;
        *) ;;
    esac
}



# Get all tracked files
git_files=$(git ls-files)

# Filter the tracked files
for f in "${ignore_files[@]}"; do
    git_files=$(grep "$f" -v <<< $git_files)
done

# Apply or update the copyright for every tracked and accepted files
for f in $git_files; do 
    # Check if the file is already copyrighted
    find_copyright=$(grep $f -c -e "${name} - Copyright (C)")
    is_copyrighted=$(($find_copyright != 0))

    filename=$(basename -- "$f")
    extension="${filename##*.}" 

    # Comment the copyright according to the file type
    commented_copyright=$(comment_copyright $extension)

    if [[ "$is_copyrighted" -ne "0" ]]; then 
        echo "$f is copyrighted : updating copyright"

        # Change (c) the old copyright lines with the new commented copyright
        sed --in-place "1,${copyright_nlines}c${commented_copyright}" $f
    else 
        echo "$f is not copyrighted : applying copyright"

        # Insert (i) the commented copyright at the 1-st line of the file
        sed --in-place "1i${commented_copyright}" $f
    fi
done
