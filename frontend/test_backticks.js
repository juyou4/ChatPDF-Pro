import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

const content = `
Check this formula:

\`L_{al} = \\frac{1}{|B|} \\Sigma_{i \\in B}\`

And this one:
$$
\\frac{1}{2}
$$
`;

async function main() {
    const file = await unified()
        .use(remarkParse)
        .use(remarkMath)
        .use(remarkRehype)
        .use(rehypeKatex)
        .use(rehypeStringify)
        .process(content);

    console.log(String(file));
}

main();
