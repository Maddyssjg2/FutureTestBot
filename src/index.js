const readline = require('readline');

const patterns = [
  {
    name: '1. Right Triangle',
    run: () => triangleRight(5),
  },
  {
    name: '2. Left Triangle',
    run: () => triangleLeft(5),
  },
  {
    name: '3. Pyramid',
    run: () => pyramid(5),
  },
  {
    name: '4. Inverted Pyramid',
    run: () => invertedPyramid(5),
  },
  {
    name: '5. Diamond',
    run: () => diamond(5),
  },
  {
    name: '6. Hollow Square',
    run: () => hollowSquare(6),
  },
  {
    name: '7. Hollow Pyramid',
    run: () => hollowPyramid(5),
  },
  {
    name: '8. Hourglass',
    run: () => hourglass(5),
  },
  {
    name: '9. Butterfly',
    run: () => butterfly(5),
  },
  {
    name: '10. X Pattern',
    run: () => xPattern(7),
  },
  {
    name: '11. Plus Pattern',
    run: () => plusPattern(7),
  },
  {
    name: '12. Right Pascal Triangle',
    run: () => pascalRight(5),
  },
  {
    name: '13. Left Pascal Triangle',
    run: () => pascalLeft(5),
  },
  {
    name: '14. Sandglass',
    run: () => sandglass(5),
  },
  {
    name: '15. Heart',
    run: () => heart(6),
  },
  {
    name: '16. Numbered Star Pyramid',
    run: () => numberedStarPyramid(5),
  },
];

function repeat(char, count) {
  return char.repeat(Math.max(0, count));
}

function triangleRight(n) {
  for (let i = 1; i <= n; i++) console.log(repeat('* ', i).trimEnd());
}

function triangleLeft(n) {
  for (let i = 1; i <= n; i++) console.log(repeat('  ', n - i) + repeat('* ', i).trimEnd());
}

function pyramid(n) {
  for (let i = 1; i <= n; i++) console.log(repeat(' ', n - i) + repeat('* ', i * 2 - 1).trimEnd());
}

function invertedPyramid(n) {
  for (let i = n; i >= 1; i--) console.log(repeat(' ', n - i) + repeat('* ', i * 2 - 1).trimEnd());
}

function diamond(n) {
  pyramid(n);
  for (let i = n - 1; i >= 1; i--) console.log(repeat(' ', n - i) + repeat('* ', i * 2 - 1).trimEnd());
}

function hollowSquare(n) {
  for (let i = 1; i <= n; i++) {
    if (i === 1 || i === n) console.log(repeat('* ', n).trimEnd());
    else console.log('*' + repeat('  ', n - 2) + ' *');
  }
}

function hollowPyramid(n) {
  for (let i = 1; i <= n; i++) {
    if (i === 1) console.log(repeat(' ', n - i) + '*');
    else if (i === n) console.log(repeat('* ', 2 * n - 1).trimEnd());
    else console.log(repeat(' ', n - i) + '*' + repeat(' ', 2 * i - 3) + '*');
  }
}

function hourglass(n) {
  invertedPyramid(n);
  for (let i = 2; i <= n; i++) console.log(repeat(' ', n - i) + repeat('* ', i * 2 - 1).trimEnd());
}

function butterfly(n) {
  for (let i = 1; i <= n; i++) console.log(repeat('* ', i).trimEnd() + repeat('  ', 2 * (n - i)) + repeat('* ', i).trimEnd());
  for (let i = n; i >= 1; i--) console.log(repeat('* ', i).trimEnd() + repeat('  ', 2 * (n - i)) + repeat('* ', i).trimEnd());
}

function xPattern(n) {
  for (let i = 1; i <= n; i++) {
    let line = '';
    for (let j = 1; j <= n; j++) line += j === i || j === n - i + 1 ? '* ' : '  ';
    console.log(line.trimEnd());
  }
}

function plusPattern(n) {
  const mid = Math.ceil(n / 2);
  for (let i = 1; i <= n; i++) {
    let line = '';
    for (let j = 1; j <= n; j++) line += i === mid || j === mid ? '* ' : '  ';
    console.log(line.trimEnd());
  }
}

function pascalRight(n) {
  for (let i = 1; i <= n; i++) console.log(repeat('* ', i).trimEnd());
}

function pascalLeft(n) {
  for (let i = 1; i <= n; i++) console.log(repeat('  ', n - i) + repeat('* ', i).trimEnd());
}

function sandglass(n) {
  invertedPyramid(n);
  pyramid(n);
}

function heart(n) {
  for (let i = n / 2; i <= n; i += 2) {
    console.log(repeat(' ', n - i) + repeat('* ', i).trimEnd() + repeat(' ', 2 * (n - i)) + repeat('* ', i).trimEnd());
  }
  for (let i = n; i >= 1; i--) {
    console.log(repeat(' ', n - i) + repeat('* ', i * 2 - 1).trimEnd());
  }
}

function numberedStarPyramid(n) {
  for (let i = 1; i <= n; i++) {
    const nums = Array.from({ length: i }, (_, idx) => idx + 1).join(' ');
    console.log(repeat(' ', n - i) + nums.split('').join(' * ') + ' *');
  }
}

function showMenu() {
  console.log('\n=== Star Patterns JS ===');
  console.log('Choose a pattern to display:');
  patterns.forEach((pattern, index) => console.log(`${index + 1}. ${pattern.name}`));
  console.log('0. Exit');
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function prompt() {
  showMenu();
  rl.question('\nEnter your choice: ', (answer) => {
    const choice = Number(answer);
    if (choice === 0) {
      console.log('Goodbye!');
      rl.close();
      return;
    }
    const selected = patterns[choice - 1];
    if (!selected) {
      console.log('Invalid choice. Please try again.');
      prompt();
      return;
    }
    console.log(`\n${selected.name}\n`);
    selected.run();
    prompt();
  });
}

prompt();
