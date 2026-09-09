function setupCopyButton() {
  const button = document.querySelector('.copy-button');
  const citation = document.getElementById('bibtex');
  const status = document.querySelector('.copy-status');
  if (!button || !citation || !status) return;

  button.addEventListener('click', async () => {
    const text = citation.textContent.trim();
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
    button.textContent = button.dataset.copiedLabel || 'Copied';
    status.textContent = status.dataset.copiedMessage || 'BibTeX copied to clipboard.';
    window.setTimeout(() => {
      button.textContent = button.dataset.defaultLabel || 'Copy BibTeX';
      status.textContent = '';
    }, 1800);
  });
}

function setupReveal() {
  const sections = document.querySelectorAll('main > section');
  if (!('IntersectionObserver' in window)) {
    sections.forEach((section) => section.classList.add('is-visible'));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px' });
  sections.forEach((section) => {
    section.classList.add('reveal-ready');
    observer.observe(section);
  });
}

function setupHeroGlow() {
  const hero = document.querySelector('.hero-block');
  if (!hero || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  hero.addEventListener('pointermove', (event) => {
    const rect = hero.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    hero.style.setProperty('--pointer-x', x + '%');
    hero.style.setProperty('--pointer-y', y + '%');
  });
}

function setupSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;

      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

function setupImageLazyLoad() {
  if ('loading' in HTMLImageElement.prototype) {
    return; // Native lazy loading is supported
  }

  const images = document.querySelectorAll('img[loading="lazy"]');
  if (!('IntersectionObserver' in window)) {
    images.forEach(img => {
      img.src = img.dataset.src || img.src;
    });
    return;
  }

  const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        if (img.dataset.src) {
          img.src = img.dataset.src;
          img.removeAttribute('data-src');
        }
        imageObserver.unobserve(img);
      }
    });
  }, {
    rootMargin: '50px 0px'
  });

  images.forEach(img => imageObserver.observe(img));
}

function setupCardAnimations() {
  const cards = document.querySelectorAll('.proof-card, .setting-card, .metric-card, .analysis-card');

  if (!('IntersectionObserver' in window)) {
    return;
  }

  const cardObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, index * 100);
        cardObserver.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px'
  });

  cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    cardObserver.observe(card);
  });
}

// Initialize all features
document.addEventListener('DOMContentLoaded', () => {
  setupCopyButton();
  setupReveal();
  setupHeroGlow();
  setupSmoothScroll();
  setupImageLazyLoad();
  setupCardAnimations();
});

// Add parallax effect to hero decorative elements
window.addEventListener('scroll', () => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const scrolled = window.pageYOffset;
  const hero = document.querySelector('.hero-block');

  if (hero && scrolled < window.innerHeight) {
    const parallaxSpeed = 0.5;
    hero.style.transform = `translateY(${scrolled * parallaxSpeed}px)`;
    hero.style.opacity = 1 - (scrolled / window.innerHeight) * 0.5;
  }
}, { passive: true });
