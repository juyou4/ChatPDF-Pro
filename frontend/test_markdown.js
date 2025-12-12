import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';

const content = `
# Math Test

Inline: $E=mc^2$

Display:
$$
\\frac{1}{2}
$$

Mixed: $a^2 + b^2 = c^2$
`;

const TestComponent = () => {
    const plugins = [
        [remarkMath, { singleDollarTextMath: true }],
        remarkGfm
    ];

    const rehypePlugins = [
        rehypeRaw,
        [rehypeKatex, { strict: false, trust: true, output: 'html' }]
    ];

    return React.createElement(ReactMarkdown, {
        remarkPlugins: plugins,
        rehypePlugins: rehypePlugins
    }, content);
};

console.log(renderToStaticMarkup(React.createElement(TestComponent)));
