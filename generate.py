import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        # Iterate through each variable in the CSP
        for variable in self.domains:
            # Filter values in the domain of the variable that have the correct length
            valid_values = {
                word for word in self.domains[variable]
                if len(word) == variable.length
            }
            # Update the domain to only include valid values
            self.domains[variable] = valid_values


    def revise(self, x, y):
        revised = False
        overlap = self.crossword.overlaps[x, y]
        
        if overlap:
            i, j = overlap
            for word_x in self.domains[x].copy():
                # Check if `word_x` aligns with at least one word in `y`'s domain
                has_overlap = any(word_x[i] == word_y[j] for word_y in self.domains[y])
                
                # If no overlap exists, remove `word_x` from `x`'s domain
                if not has_overlap:
                    self.domains[x].remove(word_x)
                    revised = True

        # Return whether a revision was made
        return revised


    def ac3(self, arcs=None):
        # Initialize arcs with all arcs in the problem if none are provided
        if not arcs:
            arcs = deque(
                (var, neighbor)
                for var in self.crossword.variables
                for neighbor in self.crossword.neighbors(var)
            )
        else:
            arcs = deque(arcs)

        # Process the arcs queue
        while arcs:
            x, y = arcs.pop()
            # If x's domain is revised, check neighboring variables
            if self.revise(x, y):
                # If x's domain becomes empty, arc consistency fails
                if len(self.domains[x]) == 0:
                    return False
                # Add neighbors of x (excluding y) back into the queue
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))

        # All arcs are consistent, return True
        return True


    def assignment_complete(self, assignment):
        # Check if every variable in the crossword has a value assigned
        return all(var in assignment for var in self.crossword.variables)


    def consistent(self, assignment):
        for var, word in assignment.items():
            # Check if the word matches the variable's length
            if len(word) != var.length:
                return False

            # Ensure all assigned words are distinct
            if list(assignment.values()).count(word) > 1:
                return False

            # Validate character consistency with neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False

        # All constraints are satisfied
        return True


    def order_domain_values(self, var, assignment):
        # Calculate how many values each word rules out for neighbors
        heuristics = {
            word: sum(
                word in self.domains[neighbor]
                for neighbor in self.crossword.neighbors(var)
                if neighbor not in assignment
            )
            for word in self.domains[var]
        }

        # Return the values sorted by the least number of values ruled out
        return sorted(heuristics, key=heuristics.get)


    def select_unassigned_variable(self, assignment):
        # Filter for unassigned variables
        unassigned = [
            var for var in self.crossword.variables if var not in assignment
        ]

        # Return the variable with the fewest remaining values in its domain,
        # breaking ties by the highest degree (number of neighbors)
        return min(
            unassigned,
            key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var)))
        )


    def backtrack(self, assignment):
        # If the assignment is complete, return it
        if self.assignment_complete(assignment):
            return assignment

        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)

        # Try each value in the variable's domain
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            # If the assignment is consistent, proceed recursively
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result:
                    return result
            # Remove the assignment if it leads to a dead end
            del assignment[var]

        # Return None if no solution is found
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
